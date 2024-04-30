# -*- coding: utf-8 -*-

import click
import sys
import os

import numpy as np
import pandas as pd
import re
import requests
from tqdm import tqdm


import torch
from torch.utils.data import TensorDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

from torch.nn.utils import clip_grad_norm_

from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from . import clf_path, config, config_path, write_default_config



model_name = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

class MNLIDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


@click.group()
def main(args=None):
    """Console script for nlp."""
    return 0

@main.command('web')
@click.option('-p', '--port', required=False, default=5000, show_default=True, help='port of web server')
def web(port):
    """
    Launch the flask web app.
    """
    from .app import app
    app.run(host='0.0.0.0', debug=True, port=port)
    

@main.command('dl-data')
def dl_data():
    """
    Download training/testing data.
    """
    # Rewrite the default configuration to make sure it's updated
    write_default_config(config_path)

    # Now reload the configuration to check the new values
    config.read(config_path)
    data_url = config.get('data', 'url')
    data_file = config.get('data', 'file')
    
    print("configuration content:")
    print("URL:", data_url)
    print("File:", data_file)

    # Proceed with the data download
    print('Downloading from %s to %s' % (data_url, data_file))
    r = requests.get(data_url)
    r.raise_for_status()  # Ensure successful request
    with open(data_file, 'wt') as f:
        f.write(r.text)


    
def load_and_tokenize_data(file_path):
    df = pd.read_csv(file_path)
    df = df[:1000]
    print("Columns in CSV:", df.columns)  # Display column names
    print("Number of rows:", len(df))  # Display number of rows


    # Check if required columns are present
    required_columns = ['premise', 'hypothesis', 'label']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise KeyError(f"Missing required columns: {', '.join(missing_columns)}")
    
    df['premise'] = df['premise'].astype(str)
    df['hypothesis'] = df['hypothesis'].astype(str)
    
    tokenized_data = tokenizer(df['premise'].tolist(), df['hypothesis'].tolist(), padding=True, truncation=True, return_tensors="pt")
    labels = torch.tensor(df['label'].values)
    
    return tokenized_data, labels

def train_model(data_file):
    # Load and tokenize the data
    df = pd.read_csv(data_file)

    df['premise'] = df['premise'].astype(str)
    df['hypothesis'] = df['hypothesis'].astype(str)

    # Tokenize the training and validation data
    train_encodings = tokenizer(
        df['premise'].tolist(),
        df['hypothesis'].tolist(),
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    labels = torch.tensor(df['label'].tolist())

    train_encodings, labels = load_and_tokenize_data(data_file)
    
    dataset = MNLIDataset(train_encodings, labels)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Training setup
    epochs = 4
    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_training_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Training loop with gradient clipping and progress bar
    model.to(device)
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}', leave=False)
        for batch in progress_bar:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss if outputs.loss is not None else loss_fn(outputs.logits, batch['labels'])
            loss.backward()

            # Gradient clipping
            clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=total_loss / len(train_loader))

        print(f'Epoch {epoch + 1}: Average Loss: {total_loss / len(train_loader)}')

    # Save the fine-tuned model and tokenizer
    save_directory = "models/fine_tuned_nli"
    os.makedirs(save_directory, exist_ok=True)
    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)

    print("Training and saving complete.")


@main.command('stats')
def stats():
    """
    Read the data files and print insightful statistics.
    """
    # Reload the configuration to ensure you get the correct data file
    config.read(config_path) 
    
    # Read the data
    data_file = config.get('data', 'file')
    df = pd.read_csv(data_file)

    # General statistics
    row_count = len(df)
    column_count = len(df.columns)

    print(f"{row_count} rows and {column_count} columns.")

    # Check for missing values
    missing_info = df.isnull().sum()
    total_missing = missing_info.sum()

    if total_missing > 0:
        print("Missing values per column:")
        print(missing_info)
    else:
        print("No missing values detected.")

    # Information about label distribution
    if 'label' in df.columns:
        label_counts = df['label'].value_counts()
        label_dist = label_counts / row_count * 100  # to get percentage
        print("Label distribution (absolute):")
        print(label_counts)
        print("Label distribution (percentage):")
        print(label_dist)
    else:
        print("No 'label' column found in the dataset.")

    # Display some sample data
    print("Sample rows:")
    print(df.head(5))  # Show the first 5 rows

    # Basic statistics for numeric columns, if any
    

@main.command('train')
def train():
    """Train the NLI classifier."""
    config.read(config_path) # Reload the configuration to get the correct file path

    # Get the file path for training data from the configuration
    data_file = config.get('data', 'file')

    if not data_file or not os.path.exists(data_file):
        raise FileNotFoundError("Training data file not found. Please run ' nlp dl-data' first.")

    # Proceed with training using the correct data file
    train_model(data_file)
    print("Training complete.")


def predict(premise, hypothesis):
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(premise, hypothesis, return_tensors="pt").to(device)
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        return predictions


@main.command('predict')
@click.argument('premise')
@click.argument('hypothesis')
def cli_predict(premise, hypothesis):
    """Make a prediction given a premise and a hypothesis."""
    prediction = predict(premise, hypothesis)
    print(f"Prediction: {prediction}")


if __name__ == "__main__":
    sys.exit(main())

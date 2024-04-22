# -*- coding: utf-8 -*-

"""Demonstrating a very simple NLP project. Yours should be more exciting than this."""
import click
import glob
import pickle
import sys
import os

import numpy as np
import pandas as pd
import re
import requests

import torch
from torch.utils.data import TensorDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from . import clf_path, config, config_path, write_default_config

model_name = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


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
    tokenized_data, labels = load_and_tokenize_data(data_file)
    dataset = TensorDataset(tokenized_data['input_ids'], tokenized_data['attention_mask'], labels)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(3):  # Example: 3 training epochs
        total_loss = 0
        for batch in train_loader:
            inputs, masks, labels = (t.to(device) for t in batch)
            model.zero_grad()
            outputs = model(inputs, attention_mask=masks, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}')



@main.command('stats')
#TODO: update stats function for my df
def stats():
    """
    Read the data files and print interesting statistics.
    """
    df = data2df()
    print('%d rows' % len(df))
    print('label counts:')
    print(df.partisan.value_counts())    

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

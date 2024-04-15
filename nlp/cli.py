# -*- coding: utf-8 -*-

"""Demonstrating a very simple NLP project. Yours should be more exciting than this."""
import click
import glob
import pickle
import sys

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

from . import clf_path, config, config_path

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
    print("Config file path:", config_path)
    config.read(config_path)  # Reload the configuration
    # data_url = config.get('data', 'url')
    # data_file = config.get('data', 'file')
    data_url = 'https://www.dropbox.com/scl/fi/8afm3cbr1ui1j3qrtv1u9/train.csv?rlkey=d0y73zduv1ira37d5xyd0sg2m&dl=0'
    data_file = '/Users/petersapountzis/.nlp/nli_train.csv'
    print('downloading from %s to %s' % (data_url, data_file))
    r = requests.get(data_url)
    with open(data_file, 'wt') as f:
        f.write(r.text)
    

def load_and_tokenize_data(file_path):
    df = pd.read_csv(file_path)
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
def stats():
    """
    Read the data files and print interesting statistics.
    """
    df = data2df()
    print('%d rows' % len(df))
    print('label counts:')
    print(df.partisan.value_counts())    

@main.command('train')
@click.argument('data_file', type=click.Path(exists=True))
def train(data_file):
    """Train the NLI classifier."""
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

import os
import json
import pickle
import boto3
import sagemaker
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.utils import to_categorical


session = sagemaker.Session()
bucket = "next-word-prediction-guvi"
s3_client = boto3.client("s3")
encoded_key = "v1/Tokinizer/encoded_text.pkl"

s3_client.download_file(bucket, encoded_key, "encoded_text.pkl")

encoded_text = None

with open("encoded_text.pkl", "rb") as file:
    encoded_text = pickle.load(file)

print("Total tokens:", len(encoded_text))


sequence_length = 50
sequences = []

for i in range(sequence_length, len(encoded_text)):
    seq = encoded_text[i-sequence_length:i+1]
    print(seq)
    sequences.append(seq)

sequences = np.array(sequences)
x = sequences[:, :-1]
y = sequences[:, -1]

vocab_size = len(set(encoded_text)) + 1
y = to_categorical(y, num_classes=vocab_size)

print("X shape:", x.shape)
print("y shape:", y.shape)

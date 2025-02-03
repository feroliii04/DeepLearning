import tensorflow as tf
import pandas as pd
from tensorflow import keras

import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt


import requests

# URLs for the data files
train_url = "https://cdn.freecodecamp.org/project-data/sms/train-data.tsv"
test_url = "https://cdn.freecodecamp.org/project-data/sms/valid-data.tsv"

# Download the train data file
train_response = requests.get(train_url)
with open("train-data.tsv", "wb") as file:
    file.write(train_response.content)

# Download the test data file
test_response = requests.get(test_url)
with open("valid-data.tsv", "wb") as file:
    file.write(test_response.content)

# Define file paths
train_file_path = "train-data.tsv"
test_file_path = "valid-data.tsv"

# Load and preprocess data
train_data = pd.read_csv(train_file_path, sep='\t', header=None, names=["label", "message"])
test_data = pd.read_csv(test_file_path, sep='\t', header=None, names=["label", "message"])

# Convert labels to binary (ham = 0, spam = 1)
train_data['label'] = train_data['label'].apply(lambda x: 1 if x == 'spam' else 0)
test_data['label'] = test_data['label'].apply(lambda x: 1 if x == 'spam' else 0)

# Tokenize the messages and pad sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Tokenizer
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(train_data['message'])

# Convert text to sequences
X_train = tokenizer.texts_to_sequences(train_data['message'])
X_test = tokenizer.texts_to_sequences(test_data['message'])

# Pad sequences
X_train = pad_sequences(X_train, padding='post', maxlen=150)
X_test = pad_sequences(X_test, padding='post', maxlen=150)

# Model Creation
model = keras.Sequential([
    keras.layers.Embedding(input_dim=10000, output_dim=128),
    keras.layers.Bidirectional(keras.layers.LSTM(128)),  # LSTM Bidireccional con más unidades
    keras.layers.Dropout(0.5),  # Dropout para evitar sobreajuste
    keras.layers.Dense(64, activation='relu'),  # Capa densa más grande
    keras.layers.Dropout(0.3),  # Dropout para evitar sobreajuste
    keras.layers.Dense(32, activation='relu'),  # Capa densa más pequeña
    keras.layers.Dense(1, activation='sigmoid')  # Capa de salida
])

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Ajustar el número de épocas y EarlyStopping
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Entrenamos el modelo
history = model.fit(X_train, train_data['label'], epochs=10, batch_size=32, validation_data=(X_test, test_data['label']), callbacks=[early_stopping])

# function to predict messages based on model
def predict_message(pred_text):
    # Convert text to sequence
    seq = tokenizer.texts_to_sequences([pred_text])
    padded = pad_sequences(seq, padding='post', maxlen=150)

    # Predict probability of being spam
    prediction_prob = model.predict(padded)[0][0]

    # Classify as spam or ham
    prediction_label = 'spam' if prediction_prob > 0.5 else 'ham'

    return [prediction_prob, prediction_label]

pred_text = "sale today! to stop texts call 98912460324"

prediction = predict_message(pred_text)
print(prediction)

# Run this cell to test your function and model. Do not modify contents.
def test_predictions():
    test_messages = ["how are you doing today",
                     "sale today! to stop texts call 98912460324",
                     "i dont want to go. can we try it a different day? available sat",
                     "our new mobile video service is live. just install on your phone to start watching.",
                     "you have won £1000 cash! call to claim your prize.",
                     "i'll bring it tomorrow. don't forget the milk.",
                     "wow, is your arm alright. that happened to me one time too"
                    ]

    test_answers = ["ham", "spam", "ham", "spam", "spam", "ham", "ham"]
    passed = True

    for msg, ans in zip(test_messages, test_answers):
        prediction = predict_message(msg)
        if prediction[1] != ans:
            passed = False

    if passed:
        print("You passed the challenge. Great job!")
    else:
        print("You haven't passed yet. Keep trying.")

test_predictions()

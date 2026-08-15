import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load dataset
df = pd.read_csv("data/symptom_nlp_dataset.csv")

tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(df["symptom_text"])

X = tokenizer.texts_to_sequences(df["symptom_text"])
X = pad_sequences(X, maxlen=10)

labels = df["label"].astype("category")
y = labels.cat.codes
label_map = dict(enumerate(labels.cat.categories))

model = Sequential([
    Embedding(5000, 64, input_length=10),
    LSTM(64),
    Dense(len(label_map), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(X, y, epochs=20, verbose=1)

model.save("model/symptom_lstm_model.h5")
pickle.dump(tokenizer, open("model/symptom_tokenizer.pkl", "wb"))
pickle.dump(label_map, open("model/symptom_label_map.pkl", "wb"))

print("✅ LSTM Symptom NLP model trained successfully")

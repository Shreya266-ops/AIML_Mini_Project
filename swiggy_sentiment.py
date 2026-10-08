import os
import re
import pickle
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# ============================================================
# 1. SETTINGS
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

DATA_PATH = "swiggy.csv"

MAX_FEATURES = 5000
MAX_LENGTH = 200

MODEL_PATH = "swiggy_sentiment_model.keras"
TOKENIZER_PATH = "swiggy_tokenizer.pkl"


# ============================================================
# 2. LOAD DATASET
# ============================================================

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"{DATA_PATH} was not found. "
        "Put swiggy.csv in the same folder as this Python file."
    )

data = pd.read_csv(DATA_PATH)

print("\nDataset shape:")
print(data.shape)

print("\nDataset columns:")
print(data.columns.tolist())

print("\nFirst 5 rows:")
print(data.head())


# ============================================================
# 3. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "Review",
    "Avg Rating"
]

for column in required_columns:
    if column not in data.columns:
        raise ValueError(
            f"Required column '{column}' is missing."
        )


# ============================================================
# 4. TEXT CLEANING
# ============================================================

def clean_text(text):
    """
    Convert review text to a simpler format.
    """

    text = str(text).lower()

    # Remove special characters
    text = re.sub(r"[^a-z0-9\s]", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


data["Review"] = data["Review"].fillna("")
data["Review"] = data["Review"].apply(clean_text)


# ============================================================
# 5. CREATE SENTIMENT LABEL
# ============================================================

# Original logic from your notebook:
# Rating > 3.5  = Positive
# Rating <= 3.5 = Negative

data["sentiment"] = (
    data["Avg Rating"] > 3.5
).astype(int)


# Remove empty reviews
data = data[
    data["Review"].str.len() > 0
].copy()

data = data.dropna(
    subset=["sentiment"]
)

data.reset_index(drop=True, inplace=True)


print("\nSentiment distribution:")
print(data["sentiment"].value_counts())


# ============================================================
# 6. VISUALIZE CLASS DISTRIBUTION
# ============================================================

class_counts = data["sentiment"].map({
    0: "Negative",
    1: "Positive"
}).value_counts()

class_counts.plot(
    kind="bar",
    title="Sentiment Distribution"
)

plt.xlabel("Sentiment")
plt.ylabel("Number of Reviews")
plt.xticks(rotation=0)

plt.tight_layout()
plt.show()


# ============================================================
# 7. TRAIN / VALIDATION / TEST SPLIT
# ============================================================

X_text = data["Review"].values
y = data["sentiment"].values


X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_text,
    y,
    test_size=0.20,
    random_state=SEED,
    stratify=y
)


X_train_text, X_val_text, y_train, y_val = train_test_split(
    X_train_text,
    y_train,
    test_size=0.10,
    random_state=SEED,
    stratify=y_train
)


print("\nDataset split:")
print("Training:", len(X_train_text))
print("Validation:", len(X_val_text))
print("Testing:", len(X_test_text))


# ============================================================
# 8. TOKENIZATION
# ============================================================

tokenizer = Tokenizer(
    num_words=MAX_FEATURES,
    oov_token="<OOV>"
)

# IMPORTANT:
# Fit tokenizer only on training data
tokenizer.fit_on_texts(X_train_text)


def encode_texts(texts):

    sequences = tokenizer.texts_to_sequences(texts)

    padded = pad_sequences(
        sequences,
        maxlen=MAX_LENGTH,
        padding="pre",
        truncating="pre"
    )

    return padded


X_train = encode_texts(X_train_text)
X_val = encode_texts(X_val_text)
X_test = encode_texts(X_test_text)


print("\nEncoded shapes:")
print("X_train:", X_train.shape)
print("X_val:", X_val.shape)
print("X_test:", X_test.shape)


# ============================================================
# 9. CREATE RNN MODEL
# ============================================================

model = Sequential([

    Embedding(
        input_dim=MAX_FEATURES,
        output_dim=32
    ),

    SimpleRNN(
        64,
        activation="tanh"
    ),

    Dropout(0.4),

    Dense(
        32,
        activation="relu"
    ),

    Dropout(0.3),

    Dense(
        1,
        activation="sigmoid"
    )
])


# ============================================================
# 10. COMPILE MODEL
# ============================================================

model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)


print("\nModel:")
model.summary()


# ============================================================
# 11. CALLBACKS
# ============================================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)


reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=1,
    min_lr=0.000001
)


# ============================================================
# 12. TRAIN MODEL
# ============================================================

history = model.fit(
    X_train,
    y_train,

    epochs=15,

    batch_size=32,

    validation_data=(
        X_val,
        y_val
    ),

    callbacks=[
        early_stopping,
        reduce_lr
    ],

    verbose=1
)


# ============================================================
# 13. TRAINING GRAPHS
# ============================================================

history_df = pd.DataFrame(
    history.history
)


# Accuracy graph

plt.figure(figsize=(8, 5))

plt.plot(
    history_df["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history_df["val_accuracy"],
    label="Validation Accuracy"
)

plt.title(
    "Training vs Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend()

plt.tight_layout()
plt.show()


# Loss graph

plt.figure(figsize=(8, 5))

plt.plot(
    history_df["loss"],
    label="Training Loss"
)

plt.plot(
    history_df["val_loss"],
    label="Validation Loss"
)

plt.title(
    "Training vs Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()

plt.tight_layout()
plt.show()


# ============================================================
# 14. TEST MODEL
# ============================================================

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("\n==============================")
print("TEST RESULTS")
print("==============================")

print(
    f"Test Loss     : {test_loss:.4f}"
)

print(
    f"Test Accuracy : {test_accuracy:.4f}"
)


# ============================================================
# 15. PREDICTIONS
# ============================================================

probabilities = model.predict(
    X_test,
    verbose=0
).ravel()


y_pred = (
    probabilities >= 0.5
).astype(int)


# ============================================================
# 16. PERFORMANCE METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)


print("\n==============================")
print("PERFORMANCE")
print("==============================")

print(
    f"Accuracy : {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall   : {recall:.4f}"
)

print(
    f"F1 Score : {f1:.4f}"
)


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Negative",
            "Positive"
        ],
        zero_division=0
    )
)


# ============================================================
# 17. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)


display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Negative",
        "Positive"
    ]
)

display.plot()

plt.title(
    "Confusion Matrix"
)

plt.tight_layout()
plt.show()


# ============================================================
# 18. PREDICT ONE REVIEW
# ============================================================

def predict_sentiment(review_text):

    cleaned_text = clean_text(
        review_text
    )

    if not cleaned_text:
        return {
            "sentiment": "Unknown",
            "confidence": 0
        }

    encoded = encode_texts(
        [cleaned_text]
    )

    probability = float(
        model.predict(
            encoded,
            verbose=0
        )[0][0]
    )


    if probability >= 0.5:

        sentiment = "Positive"

        confidence = probability

    else:

        sentiment = "Negative"

        confidence = 1 - probability


    return {
        "sentiment": sentiment,
        "confidence": confidence,
        "positive_probability": probability
    }


# ============================================================
# 19. TEST CUSTOM REVIEW
# ============================================================

sample_review = (
    "The food was delicious and delivery was very fast."
)

result = predict_sentiment(
    sample_review
)

print("\n==============================")
print("CUSTOM REVIEW")
print("==============================")

print(
    "Review:",
    sample_review
)

print(
    "Prediction:",
    result
)


# ============================================================
# 20. BATCH PREDICTION
# ============================================================

def predict_reviews(reviews):

    cleaned_reviews = [
        clean_text(review)
        for review in reviews
    ]

    encoded = encode_texts(
        cleaned_reviews
    )

    probabilities = model.predict(
        encoded,
        verbose=0
    ).ravel()


    results = pd.DataFrame({

        "Review": reviews,

        "Positive Probability":
            probabilities,

        "Sentiment":
            np.where(
                probabilities >= 0.5,
                "Positive",
                "Negative"
            ),

        "Confidence":
            np.where(
                probabilities >= 0.5,
                probabilities,
                1 - probabilities
            )
    })


    return results


reviews = [

    "Amazing food and excellent service.",

    "Very bad experience. Food was cold.",

    "The delivery was fast and the food was good."

]


results = predict_reviews(
    reviews
)

print("\nBatch predictions:")

print(results)


# ============================================================
# 21. SAVE MODEL
# ============================================================

model.save(
    MODEL_PATH
)


# Save tokenizer

with open(
    TOKENIZER_PATH,
    "wb"
) as file:

    pickle.dump(
        tokenizer,
        file
    )


print("\nModel saved as:")
print(MODEL_PATH)

print("\nTokenizer saved as:")
print(TOKENIZER_PATH)


# ============================================================
# 22. LOAD MODEL AGAIN
# ============================================================

loaded_model = load_model(
    MODEL_PATH
)


with open(
    TOKENIZER_PATH,
    "rb"
) as file:

    loaded_tokenizer = pickle.load(
        file
    )


print("\nModel and tokenizer loaded successfully.")


# ============================================================
# 23. FINAL TEST
# ============================================================

test_review = (
    "The restaurant was excellent and the food tasted great."
)

sequence = loaded_tokenizer.texts_to_sequences(
    [clean_text(test_review)]
)

padded = pad_sequences(
    sequence,
    maxlen=MAX_LENGTH
)

probability = float(
    loaded_model.predict(
        padded,
        verbose=0
    )[0][0]
)


sentiment = (
    "Positive"
    if probability >= 0.5
    else "Negative"
)


print("\n==============================")
print("FINAL PREDICTION")
print("==============================")

print(
    "Review:",
    test_review
)

print(
    "Sentiment:",
    sentiment
)

print(
    "Probability:",
    round(probability, 3)
)
import pandas as pd
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
from tqdm import tqdm

# Device configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load GPT-2 model and tokenizer
model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

THRESHOLD = 0.5


def calculate_perplexity(text):
    """Calculate perplexity score for a given text."""
    try:
        encodings = tokenizer(
            text,
            return_tensors='pt',
            max_length=1024,
            truncation=True
        ).to(device)

        with torch.no_grad():
            outputs = model(**encodings, labels=encodings.input_ids)

        return torch.exp(outputs.loss).item()

    except Exception as e:
        print(f"Error processing text: {e}")
        return np.nan


def train_classifier():
    # Load datasets
    human_df = pd.read_csv('DATASET_AD - Human.csv', encoding='utf-8')
    ai_df = pd.read_csv('DATASET_AD_AI_Updated.csv', encoding='utf-8')

    # Assign labels
    human_df['label'] = 0
    ai_df['label'] = 1

    # Combine datasets
    df = pd.concat([human_df, ai_df], ignore_index=True)

    # Clean text column
    df['IEEE'] = df['IEEE'].fillna('').astype(str)

    # Remove empty rows
    df = df[df['IEEE'].str.strip() != ''].copy()

    print(f"\nTotal valid samples: {len(df)}")
    print("\nCalculating perplexity scores...")

    # Calculate perplexity safely
    perplexities = []

    for text in tqdm(df['IEEE'], desc="Processing"):
        score = calculate_perplexity(text)
        perplexities.append(score)

    df['perplexity'] = perplexities

    # Remove failed rows
    df = df.dropna(subset=['perplexity'])

    # Log transform to handle outliers
    df['perplexity'] = df['perplexity'].apply(np.log1p)

    # Prepare data
    X = df[['perplexity']].values
    y = df['label'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Train classifier
    clf = LogisticRegression(
        class_weight='balanced',
        max_iter=1000
    )

    clf.fit(X_train, y_train)

    # Save model
    joblib.dump(clf, 'ai_detector_clf.pkl')

    # Evaluate model
    y_pred = clf.predict(X_test)

    print(f"\nModel Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    print("\nModel saved successfully as ai_detector_clf.pkl")


def load_classifier():
    return joblib.load('ai_detector_clf.pkl')


if __name__ == "__main__":
    train_classifier()

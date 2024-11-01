import kagglehub
from transformers import AutoTokenizer
import jax.numpy as jnp
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(model_name="bert-base-uncased", max_length=100):
    # Download the IMDb dataset
    path = kagglehub.dataset_download(
        "lakshmi25npathi/imdb-dataset-of-50k-movie-reviews"
    )

    file = path + "/IMDB Dataset.csv"
    dataset = pd.read_csv(file)
    dataset["sentiment"] = dataset["sentiment"].replace({"positive": 1, "negative": 0})
    train_data, test_data = train_test_split(
        dataset, test_size=0.2, stratify=dataset["sentiment"], random_state=42
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_batch(reviews, batch):
        tokens = tokenizer(
            reviews,
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        return {
            "inputs": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
            "labels": batch,
        }

    tokenized_train_dataset = tokenize_batch(
        train_data["review"].tolist(), train_data["sentiment"].tolist()
    )
    train_inputs = jnp.array(tokenized_train_dataset["inputs"])
    train_labels = jnp.array(tokenized_train_dataset["labels"])

    unique, counts = np.unique(train_labels, return_counts=True)
    print("Training label distribution:", dict(zip(unique, counts)))

    tokenized_test_dataset = tokenize_batch(
        test_data["review"].tolist(), test_data["sentiment"].tolist()
    )
    test_inputs = jnp.array(tokenized_test_dataset["inputs"])
    test_labels = jnp.array(tokenized_test_dataset["labels"])

    # Check distribution in testing data
    unique, counts = np.unique(test_labels, return_counts=True)
    print("Testing label distribution:", dict(zip(unique, counts)))

    return (train_inputs, train_labels), (test_inputs, test_labels)


load_data()

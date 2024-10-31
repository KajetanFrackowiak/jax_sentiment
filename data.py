from transformers import load_dataset
import jax.numpy as jnp


def load_data():
    dataset = load_dataset("imdb")
    texts = dataset["train"]["text"]
    labels = dataset["train"]["label"]
    labels = ["positive" if label == 1 else "negative" for label in labels]
    return texts, jnp.array(labels)

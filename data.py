from datasets import load_dataset
from transformers import AutoTokenizer
import jax.numpy as jnp


def load_data(model_name="bert-base-uncased", max_length=100):
    dataset = load_dataset("imdb")
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    unique_train_labels = set(train_dataset["label"])
    unique_test_labels = set(test_dataset["label"])

    print("Unique train labels:", len(unique_train_labels))
    print("Unique test labels:", len(unique_test_labels))

    def tokenize_batch(batch):
        tokens = tokenizer(
            batch["text"], padding="max_length", truncation=True, max_length=max_length
        )
        return {"inputs": tokens["input_ids"], "labels": batch["label"]}

    tokenized_train_dataset = dataset["train"].map(tokenize_batch, batched=True)
    tokenized_test_dataset = dataset["test"].map(tokenize_batch, batched=True)

    # Extract inputs and labels as JAX arrays
    train_inputs = jnp.array(tokenized_train_dataset["inputs"])
    train_labels = jnp.array([1 if label == "pos" else 0 for label in tokenized_train_dataset["labels"]])

    test_inputs = jnp.array(tokenized_test_dataset["inputs"])
    test_labels = jnp.array([1 if label == "pos" else 0 for label in tokenized_test_dataset["labels"]])


    print("Sample trian inputs:", train_inputs[:3])
    print("Sample trian labels:", train_labels[:3])
    print("Sample test inputs:", test_inputs[:3])
    print("Sample test labels:", test_labels[:3])

    print("Train label distribution", {0: jnp.sum(train_labels == 0), 1: jnp.sum(train_labels == 1)})
    print("Test label distribution", {0: jnp.sum(test_labels == 0), 1: jnp.sum(test_labels == 1)})

    print("Train input shape:", train_inputs.shape)
    print("Test input shape:", test_inputs.shape)
    print("Train label shape:", train_labels.shape)
    print("Test label shape:", test_labels.shape)

    return (train_inputs, train_labels), (test_inputs, test_labels)

load_data()
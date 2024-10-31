from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import jax
import optax
from jax import numpy as jnp
from transformers import Trainer, TrainingArguments

# Load the IMDb dataset
dataset = load_dataset("imdb")

# Check the dataset structure
print(dataset)

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


# Preprocessing function
def preprocess_function(examples):
    return {
        "input_ids": tokenizer(
            examples["text"], truncation=True, padding="max_length", max_length=512
        )["input_ids"],
        "labels": examples["label"],
    }


# Preprocess the datasets
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Load the model
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2
)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    num_train_epochs=3,
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
)

# Train the model
trainer.train()

# Evaluate the model
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")


# JIT-compile the prediction function
@jax.jit
def predict_jit(model, inputs):
    logits = model(**inputs).logits
    predicted_class = jnp.argmax(logits, axis=-1)
    return predicted_class


# Function to make predictions
def predict(text):
    inputs = tokenizer(
        text,
        return_tensors="jax",
        truncation=True,
        padding="max_length",
        max_length=512,
    )
    # Use the JIT-compiled prediction
    predicted_class = predict_jit(model, inputs)
    return predicted_class


# Example prediction
print(predict("I love this movie!"))

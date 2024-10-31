import gradio as gr
import jax
import jax.numpy as jnp
import numpy as np
from model import SentimentModel
from transformers import AutoTokenizer

def load_model_params():
    with np.load("model_params.npz", allow_pickle=True) as data:
        return data["params"].item()


model = SentimentModel(num_classes=2)
params = load_model_params()

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

def predict(text):
    tokenized_input = tokenizer(text, padding="max_length", truncation=True, max_length=100, return_tensors="jax")
    inputs = tokenized_input["input_ids"]
    logits = model.apply({"params": params}, inputs)
    prediction = jnp.argmax(logits, axis=-1)
    # return "positive" if prediction == 1 else "negative"
    return prediction

iface = gr.Interface(
    fn=predict,
    inputs="text",
    outputs="text",
    title="Sentiment Analysis",
    description="Enter reviews to analyze their sentiment",
    allow_flagging="never",
)

if __name__ == "__main__":
    sample_text = "I love the movies"
    print(predict(sample_text))
    # iface.launch()
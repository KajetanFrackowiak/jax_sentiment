import gradio as gr
import jax
import jax.numpy as jnp
import numpy as np
from model import SentimentModel
from transformers import AutoTokenizer, AutoModelForAudioClassification


def load_model_params():
    with np.load("model_params.npz") as data:
        return data["params"]


model = SentimentModel(num_classes=2)
params = load_model_params()


def predict(text):
    inputs = jnp.random.rand(1, 100)
    logits = model.apply({"params": params}, inputs)
    prediction = jnp.argmax(logits, axis=-1)
    output_labels = ["positive" if pred == 1 else "negative" for pred in prediction]
    return output_labels


iface = gr.Interface(
    fn=predict,
    inputs="text",
    outputs="text",
    title="Sentiment Analysis",
    description="Enter reviews to analyze their sentiment",
    allow_flagging="never",
)

iface

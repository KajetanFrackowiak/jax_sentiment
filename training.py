import jax
import jax.numpy as jnp
from flax.training import train_state
from model import SentimentModel
from data import load_data
import optax
from tqdm import tqdm


@jax.jit
def train_step(state: train_state.TrainState, batch):
    def loss_fn(params):
        logits = state.apply_fn({"params": params}, batch["inputs"])
        loss = jax.nn.softmax_cross_entropy(logits=logits, labels=batch["labels"])
        return loss.mean()

    grads = jax.grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)


@jax.jit
def eval_step(state: train_state.TrainState, batch):
    logits = state.apply_fn({"params": state.params}, batch["labels"])
    return jnp.argmax(logits, axis=-1)


def main():
    texts, labels = load_data()

    model = SentimentModel(num_classes=2)
    rng = jax.random.PRNGKey(0)
    params = model.init(rng, jnp.ones((1, 100)))["params"]

    optimizer = optax.adam(learning_rate=1e-3)
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=optimizer
    )

    num_epochs = 10

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        batch = {"inputs": texts, "labels": labels}
        state = train_step(state, batch)

        with open("model_params.npz", "wb") as f:
            jax.numpy.savez(f, params=state.params)


if __name__ == "__main__":
    main()

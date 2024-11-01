import jax
import jax.numpy as jnp
from flax.training import train_state
from model import SentimentModel
from data import load_data
import optax
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def one_hot(labels, num_classes):
    return jax.nn.one_hot(labels, num_classes)


def calculate_accuracy(predictions, labels):
    correct_predictions = jnp.sum(predictions == labels)
    total_predictions = labels.shape[0]
    accuracy = correct_predictions / total_predictions
    return accuracy


@jax.jit
def train_step(state: train_state.TrainState, batch, rng):
    def loss_fn(params):
        logits = state.apply_fn({"params": params}, batch["inputs"], rng=rng)
        loss = optax.softmax_cross_entropy(logits=logits, labels=batch["labels"])
        return loss.mean()

    # Split the RNG for dropout layers
    rng, rng_key = jax.random.split(rng)

    grads = jax.grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    loss_value = loss_fn(state.params)

    # Update the RNG key
    rng = rng_key

    # Evaluate accuracy on the training batch
    logits = state.apply_fn(
        {"params": state.params}, batch["inputs"], rng=rng
    )  # Pass the RNG here
    predictions = jnp.argmax(logits, axis=-1)
    labels = jnp.argmax(batch["labels"], axis=-1)
    accuracy = calculate_accuracy(predictions, labels)

    return state, loss_value, accuracy, rng


@jax.jit
def eval_step(state: train_state.TrainState, batch, rng):
    # Split the RNG for evaluation
    rng, rng_key = jax.random.split(rng)
    logits = state.apply_fn(
        {"params": state.params}, batch["inputs"], rng=rng_key
    )  # Pass the RNG here
    return jnp.argmax(logits, axis=-1), rng


def evaluate_model(state, texts, labels, rng):
    predictions, rng = eval_step(
        state, {"inputs": texts}, rng
    )  # Pass the RNG to eval_step
    accuracy = calculate_accuracy(predictions, jnp.argmax(labels, axis=-1))
    return accuracy, rng


def print_predictions(state, inputs, labels, rng):
    predictions, rng = eval_step(
        state, {"inputs": inputs}, rng
    )  # Pass the RNG to eval_step
    correct = predictions == labels
    for i in range(5):
        print(f"Input: {inputs[i]}, Predicted: {predictions[i]}, Correct: {correct[i]}")


def plot_confusion_matrix(state, inputs, labels, rng):
    predictions, rng = eval_step(
        state, {"inputs": inputs}, rng
    )  # Pass the RNG to eval_step
    cm = confusion_matrix(labels, predictions)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Negative", "Positive"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()


def create_train_state(rng, model, num_classes):
    """Creates initial `TrainState`."""
    params = model.init(rng, jnp.ones((1, 100)))[
        "params"
    ]  # Adjust input shape as necessary
    return train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=optax.adam(learning_rate=1e-3)
    )


def main():
    (train_inputs, train_labels), (test_inputs, test_labels) = load_data()

    # One-hot encode labels
    labels_one_hot = one_hot(train_labels, num_classes=2)

    model = SentimentModel(num_classes=2)
    rng = jax.random.PRNGKey(0)
    state = create_train_state(rng, model, num_classes=2)

    num_epochs = 10
    losses = []
    accuracies = []
    test_accuracies = []

    for epoch in tqdm(range(num_epochs), desc="Training Epochs", unit="epoch"):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        batch = {"inputs": train_inputs, "labels": labels_one_hot}
        state, loss_value, accuracy, rng = train_step(
            state, batch, rng
        )  # Pass rng to train_step
        losses.append(loss_value)
        accuracies.append(accuracy)

        test_labels_one_hot = one_hot(test_labels, num_classes=2)
        test_accuracy, rng = evaluate_model(
            state, test_inputs, test_labels_one_hot, rng
        )  # Pass rng to evaluate_model
        test_accuracies.append(test_accuracy)

        print(
            f"Loss: {loss_value:.4f}, Accuracy: {accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}"
        )

        # Save model parameters
        with open("model_params.npz", "wb") as f:
            jax.numpy.savez(f, params=state.params)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.subplot(1, 2, 2)
    plt.plot(accuracies, label="Training Accuracy", color="blue")
    plt.plot(test_accuracies, label="Test Accuracy", color="red")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

    print_predictions(
        state, test_inputs, jnp.argmax(test_labels_one_hot, axis=-1), rng
    )  # Pass rng to print_predictions
    plot_confusion_matrix(
        state, test_inputs, jnp.argmax(test_labels_one_hot, axis=-1), rng
    )  # Pass rng to plot_confusion_matrix


if __name__ == "__main__":
    main()

import flax.linen as nn
import jax.numpy as jnp
import jax.random


class SentimentModel(nn.Module):
    num_classes: int

    def setup(self):
        self.dense1 = nn.Dense(16, kernel_init=nn.initializers.xavier_uniform())

        self.dense2 = nn.Dense(
            self.num_classes, kernel_init=nn.initializers.xavier_uniform()
        )

    def __call__(self, x, train: bool = True, rng: jnp.ndarray = None):
        x = self.dense1(x)
        x = nn.relu(x)

        x = self.dense2(x)
        return x

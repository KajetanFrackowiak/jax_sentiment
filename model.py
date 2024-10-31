import flax.linen as nn
import jax.numpy as jnp
import jax.random


class SentimentModel(nn.Module):
    num_classes: int

    def setup(self):
        self.dense1 = nn.Dense(32, kernel_init=nn.initializers.xavier_uniform())
        self.droupout1 = nn.Dropout(0.2)
        self.dense2 = nn.Dense(64, kernel_init=nn.initializers.xavier_uniform())
        self.droupout2 = nn.Dropout(0.2)
        self.dense3 = nn.Dense(
            self.num_classes, kernel_init=nn.initializers.xavier_uniform()
        )

    def __call__(self, x, train: bool = True, rng: jnp.ndarray = None):
        x = self.dense1(x)
        x = nn.relu(x)
        x = self.droupout1(x, deterministic=not train, rng=rng)

        x = self.dense2(x)
        x = nn.relu(x)
        x = self.droupout2(x, deterministic=not train, rng=rng)

        x = self.dense3(x)
        return x

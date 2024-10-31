import flax.linen as nn


class SentimentModel(nn.Module):
    num_classes: int

    def setup(self):
        self.dense1 = nn.Dense(128, kernel_init=nn.initializers.xavier_uniform())
        self.dense2 = nn.Dense(
            self.num_classes, kernel_init=nn.initializers.xavier_uniform()
        )

    def __call__(self, x):
        x = nn.relu(self.dense1(x))
        return self.dense2(x)

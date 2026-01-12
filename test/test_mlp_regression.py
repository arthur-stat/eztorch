import os
import sys

from eztorch.layers.sequential import Sequential

ROOT = os.path.dirname(os.path.dirname(__file__))
SRC_DIR = os.path.join(ROOT, "src")
TEST_DIR = os.path.dirname(__file__)
for path in (SRC_DIR, TEST_DIR):
    if path not in sys.path:
        sys.path.append(path)

import numpy as np
import matplotlib.pyplot as plt

from eztorch.layers.linear import Linear
from eztorch.functions.activations import Sigmoid
from eztorch.optim.adam import Adam
from eztorch.utils.trainer import Trainer
from eztorch.functions.losses import MSELoss
from utils import configure_plot_style, plot_loss_curve, plot_regression_fit


def main():
    configure_plot_style()

    rng = np.random.default_rng(0)
    n_samples = 500
    X = rng.uniform(-2.0, 2.0, size=(n_samples, 1))
    noise = rng.normal(0.0, 0.3, size=(n_samples, 1))
    y = 3.0 * X - 1.0 + noise

    batch_size = 50
    max_steps = 800
    log_every = 100
    learning_rate = 0.05

    model = Sequential([
        Linear(1, 16), Sigmoid(),
        Linear(16, 1)
    ])
    optimizer = Adam(lr=learning_rate)
    trainer = Trainer(model=model, forward=model, optimizer=optimizer, loss_fn=MSELoss())

    loss_record = trainer.fit(X, y, batch_size=batch_size, max_steps=max_steps, log_every=log_every)

    plot_loss_curve(loss_record, step_interval=log_every, title="Regression Training Loss")
    # plt.savefig("regression_training_loss.svg", format="svg")
    plt.show()

    plot_regression_fit(X, y, model, title="Regression Fit")
    plt.tight_layout()
    # plt.savefig("regression_fit.svg", format="svg")
    plt.show()


if __name__ == "__main__":
    main()
import os
import sys

from eztorch.layers.sequential import Sequential

ROOT = os.path.dirname(os.path.dirname(__file__))
SRC_DIR = os.path.join(ROOT, "src")
TEST_DIR = os.path.dirname(__file__)
for path in (SRC_DIR, TEST_DIR):
    if path not in sys.path:
        sys.path.append(path)

import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

from eztorch.models.sequential_model import SequentialModel
from eztorch.optim.adam import Adam
from eztorch.utils.trainer import Trainer
from eztorch.layers.linear import Linear
from eztorch.layers.norm import BatchNorm1d, LayerNorm
from eztorch.functions.activations import ReLU
from utils import configure_plot_style, plot_loss_curve, scatter_classes, plot_decision_boundary


def main():
    configure_plot_style()

    X, y = make_moons(n_samples=1000, noise=0.1)
    batch_size = 50
    max_steps = 1000
    log_every = 100
    learning_rate = 0.05

    mlp = SequentialModel(Sequential([
        Linear(2, 8), BatchNorm1d(8), ReLU(),
        Linear(8, 8), LayerNorm(8), ReLU(),
        Linear(8, 2)
    ]))
    optimizer = Adam(lr=learning_rate)
    trainer = Trainer(model=mlp.model, forward=mlp.forward, optimizer=optimizer)

    loss_record = trainer.fit(X, y, batch_size=batch_size, max_steps=max_steps, log_every=log_every)

    plot_loss_curve(loss_record, step_interval=log_every, title="Training Loss Curve")
    # plt.savefig("training_loss_curve.svg", format="svg")
    plt.show()

    ax_data = scatter_classes(X, y, title="Data Points Visualization")
    plt.tight_layout()
    # plt.savefig("data_distribution.svg", format="svg")
    plt.show()

    ax_boundary = scatter_classes(X, y, title="Decision Boundary")
    plot_decision_boundary(mlp, ax_boundary)
    plt.tight_layout()
    # plt.savefig("hyperplane.svg", format="svg")
    plt.show()


if __name__ == "__main__":
    main()
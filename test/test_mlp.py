import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
SRC_DIR = os.path.join(ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

from eztorch.models.mlp import MLP
from eztorch.optim.adam import Adam
from eztorch.engine.trainer import Trainer


def main():

    X, y = make_moons(n_samples=1000, noise=0.1)
    batch_size = 50
    max_steps = 50000
    learning_rate = 0.05

    mlp = MLP([
        Linear(2, 4), Sigmoid(),
        Linear(4, 4), Sigmoid(),
        Linear(4, 2)
    ])
    optimizer = Adam(lr=learning_rate)
    trainer = Trainer(model=mlp.model, forward=mlp.forward, optimizer=optimizer)

    lossRecord = trainer.fit(X, y, batch_size=batch_size, max_steps=max_steps, log_every=1000)

    steps = np.arange(0, max_steps, 1000)
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'lines.linewidth': 2,
        'lines.markersize': 5,
        'figure.figsize': (8, 6),
        'axes.grid': True,
        'grid.alpha': 0.5,
        'grid.linestyle': '--',
        'grid.color': 'gray'
    })
    plt.plot(steps, lossRecord, color='#9b95c9', label='Training Loss')
    plt.title('Training Loss Curve', fontsize=16)
    plt.xlabel('Steps', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(loc='upper right', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    # plt.savefig("training_loss_curve.svg", format="svg")
    plt.show()

    # Visualization helpers (reused)
    def draw_data(data):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(1, 1, 1)
        x, y = data
        label1 = x[y > 0]
        ax.scatter(label1[:, 0], label1[:, 1], marker='o', color='royalblue', label='Class 1', alpha=0.7)
        label0 = x[y == 0]
        ax.scatter(label0[:, 0], label0[:, 1], marker='^', color='darkorange', label='Class 0', alpha=0.7)
        ax.set_title('Data Points Visualization', fontsize=16)
        ax.set_xlabel('Feature 1', fontsize=14)
        ax.set_ylabel('Feature 2', fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.legend(loc='upper right', fontsize=12)
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
        return ax

    def draw_model(ax, model):
        x1 = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 100)
        x2 = np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 100)
        x1, x2 = np.meshgrid(x1, x2)
        y = model.probability(np.c_[x1.ravel(), x2.ravel()])[:, 1]
        y = y.reshape(x1.shape)
        ax.contourf(x1, x2, y, levels=[0, 0.5], colors=['#9B95C9'], alpha=0.15)
        ax.contour(x1, x2, y, levels=[0.5], colors='#7B68EE', linewidths=2, linestyles='solid')
        ax.set_title('Decision Boundary', fontsize=16)
        ax.set_xlabel('Feature 1', fontsize=14)
        ax.set_ylabel('Feature 2', fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
        return ax

    draw_data([X, y])
    plt.tight_layout()
    # plt.savefig("data_distribution.svg", format="svg")
    plt.show()

    draw_model(draw_data([X, y]), mlp)
    plt.tight_layout()
    # plt.savefig("hyperplane.svg", format="svg")
    plt.show()


if __name__ == "__main__":
    main()
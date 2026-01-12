"""
This plotting code is adapted from 唐亘's work. Many thanks!
 - 唐亘's GitHub: https://github.com/GenTang
 - The monograph 唐亘 authored: https://github.com/GenTang/regression2chatgpt

I first came across 唐亘’s videos towards the end of 2024, just after I had been admitted as a master’s student.
Now it is already 2026—how swiftly time passes.
"""
import numpy as np
import matplotlib.pyplot as plt


def configure_plot_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "lines.linewidth": 2,
        "lines.markersize": 5,
        "figure.figsize": (8, 6),
        "axes.grid": True,
        "grid.alpha": 0.5,
        "grid.linestyle": "--",
        "grid.color": "gray",
    })


def plot_loss_curve(loss_record: list[float], step_interval: int, title: str = "Training Loss Curve"):
    steps = np.arange(0, step_interval * len(loss_record), step_interval)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(steps, loss_record, color="#9b95c9", label="Training Loss")
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Steps", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.legend(loc="upper right", fontsize=12)
    ax.tick_params(axis="both", labelsize=10)
    fig.tight_layout()
    return ax


def scatter_classes(X: np.ndarray, y: np.ndarray, ax=None, title: str = "Data Points Visualization"):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(X[y > 0][:, 0], X[y > 0][:, 1], marker="o", color="royalblue", label="Class 1", alpha=0.7)
    ax.scatter(X[y == 0][:, 0], X[y == 0][:, 1], marker="^", color="darkorange", label="Class 0", alpha=0.7)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Feature 1", fontsize=14)
    ax.set_ylabel("Feature 2", fontsize=14)
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.legend(loc="upper right", fontsize=12)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    return ax


def plot_decision_boundary(model, ax, resolution: int = 100, title: str = "Decision Boundary"):
    x1 = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], resolution)
    x2 = np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], resolution)
    x1_grid, x2_grid = np.meshgrid(x1, x2)
    probs = model.probability(np.c_[x1_grid.ravel(), x2_grid.ravel()])[:, 1]
    probs = probs.reshape(x1_grid.shape)
    ax.contourf(x1_grid, x2_grid, probs, levels=[0, 0.5], colors=["#9B95C9"], alpha=0.15)
    ax.contour(x1_grid, x2_grid, probs, levels=[0.5], colors="#7B68EE", linewidths=2, linestyles="solid")
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Feature 1", fontsize=14)
    ax.set_ylabel("Feature 2", fontsize=14)
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    return ax


def plot_regression_fit(X: np.ndarray, y_true: np.ndarray, model, ax=None, title: str = "Regression Fit"):
    x_flat = X.squeeze()
    y_flat = y_true.squeeze()
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    order = np.argsort(x_flat)
    y_pred = model(X).squeeze()
    ax.scatter(x_flat, y_flat, color="darkorange", alpha=0.6, label="Targets")
    ax.plot(x_flat[order], y_pred[order], color="royalblue", label="Predictions")
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Input", fontsize=14)
    ax.set_ylabel("Output", fontsize=14)
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.legend(loc="upper right", fontsize=12)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    return ax
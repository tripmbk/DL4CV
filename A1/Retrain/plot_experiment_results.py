
import json
import os
import re
import matplotlib.pyplot as plt
import numpy as np


def plot_experiment_results(
    json_path="results/experiment_results.json",
    output_dir="results/plots",
    experiment_ids=None
):
    os.makedirs(output_dir, exist_ok=True)

    # =================================================
    # 1. READ JSON
    # =================================================

    if not os.path.isfile(json_path):
        raise FileNotFoundError(
            f"Không tìm thấy file: {os.path.abspath(json_path)}"
        )

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    experiments = data.get("experiments", [])

    # Filter by experiment IDs if requested
    if experiment_ids is not None:
        experiments = [
            exp for exp in experiments
            if exp.get("experiment_id") in experiment_ids
        ]

    # Exclude failed experiments and experiments without history
    valid_experiments = []

    for exp in experiments:
        status = exp.get("metadata", {}).get("status", "unknown")
        history = exp.get("training_history", [])

        if status == "failed":
            print("Skip failed experiment:", exp.get("experiment_id"))
            continue

        if not history:
            print("Skip empty history:", exp.get("experiment_id"))
            continue

        valid_experiments.append(exp)

    experiments = valid_experiments

    if not experiments:
        print("Không tìm thấy experiment hợp lệ.")
        return

    print(f"Tìm thấy {len(experiments)} experiment hợp lệ.")

    # =================================================
    # 2. HELPER FUNCTIONS
    # =================================================

    def get_name(exp):
        return exp.get("experiment_id", "Unknown")

    def safe_filename(name):
        return re.sub(r'[\\/*?:"<>|]', "_", name)

    def get_numeric_value(value):
        if value is None:
            return None

        try:
            value = float(value)
            return value if np.isfinite(value) else None
        except (TypeError, ValueError):
            return None

    # Convert accuracy from fraction [0, 1] to percentage [0, 100]
    def normalize_accuracy(value):
        value = get_numeric_value(value)

        if value is None:
            return None

        if 0 <= value <= 1:
            return value * 100.0

        return value

    def get_memory(computation):
        # Support different logger field names
        value = computation.get(
            "max_peak_memory_allocated_mb"
        )

        if value is None:
            value = computation.get(
                "peak_memory_allocated_mb"
            )

        return get_numeric_value(value)

    # =================================================
    # 3. LEARNING CURVES
    # =================================================

    for exp in experiments:

        name = get_name(exp)
        filename = safe_filename(name)

        history = exp.get("training_history", [])

        epochs = [
            item.get("epoch")
            for item in history
        ]

        train_loss = [
            get_numeric_value(item.get("train_loss"))
            for item in history
        ]

        val_loss = [
            get_numeric_value(item.get("val_loss"))
            for item in history
        ]

        train_acc = [
            normalize_accuracy(item.get("train_accuracy"))
            for item in history
        ]

        val_acc = [
            normalize_accuracy(item.get("val_accuracy"))
            for item in history
        ]

        # -------------------------------------------------
        # LOSS CURVE
        # -------------------------------------------------

        plt.figure(figsize=(10, 5))

        plt.plot(
            epochs,
            train_loss,
            label="Train Loss",
            marker="o",
            markersize=3
        )

        plt.plot(
            epochs,
            val_loss,
            label="Validation Loss",
            marker="s",
            markersize=3
        )

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Loss Curve - {name}")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

        plt.savefig(
            os.path.join(
                output_dir,
                f"{filename}_loss.png"
            ),
            dpi=300
        )

        plt.close()

        # -------------------------------------------------
        # ACCURACY CURVE
        # -------------------------------------------------

        plt.figure(figsize=(10, 5))

        plt.plot(
            epochs,
            train_acc,
            label="Train Accuracy",
            marker="o",
            markersize=3
        )

        plt.plot(
            epochs,
            val_acc,
            label="Validation Accuracy",
            marker="s",
            markersize=3
        )

        # Best checkpoint
        best = exp.get("best_checkpoint", {})

        best_epoch = best.get("best_epoch")
        best_accuracy = normalize_accuracy(
            best.get("best_val_accuracy")
        )

        if best_epoch is not None and best_accuracy is not None:

            plt.scatter(
                [best_epoch],
                [best_accuracy],
                color="red",
                marker="*",
                s=180,
                label=f"Best Val Acc: {best_accuracy:.2f}%"
            )

        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.title(f"Accuracy Curve - {name}")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

        plt.savefig(
            os.path.join(
                output_dir,
                f"{filename}_accuracy.png"
            ),
            dpi=300
        )

        plt.close()

        print(f"Created learning curves: {name}")

    # =================================================
    # 4. COMPARISON FUNCTION
    # =================================================

    def plot_comparison(
        title,
        ylabel,
        filename,
        value_function,
        normalize=False
    ):

        names = []
        values = []

        for exp in experiments:

            value = value_function(exp)

            if normalize:
                value = normalize_accuracy(value)
            else:
                value = get_numeric_value(value)

            if value is not None:
                names.append(get_name(exp))
                values.append(value)

        if not values:
            print(f"No data for: {title}")
            return

        plt.figure(figsize=(12, 6))

        bars = plt.bar(names, values)

        plt.title(title)
        plt.xlabel("Experiment")
        plt.ylabel(ylabel)

        plt.xticks(rotation=30, ha="right")
        plt.grid(axis="y", alpha=0.3)

        for bar, value in zip(bars, values):

            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.2f}",
                ha="center",
                va="bottom"
            )

        plt.tight_layout()

        plt.savefig(
            os.path.join(output_dir, filename),
            dpi=300
        )

        plt.close()

        print("Created comparison:", filename)

    # =================================================
    # 5. TEST ACCURACY
    # =================================================

    plot_comparison(
        title="Test Accuracy Comparison",
        ylabel="Accuracy (%)",
        filename="test_accuracy.png",
        value_function=lambda exp:
            exp.get("testing", {}).get("accuracy"),
        normalize=True
    )

    # =================================================
    # 6. BEST VALIDATION ACCURACY
    # =================================================

    plot_comparison(
        title="Best Validation Accuracy Comparison",
        ylabel="Accuracy (%)",
        filename="best_val_accuracy.png",
        value_function=lambda exp:
            exp.get("best_checkpoint", {}).get(
                "best_val_accuracy"
            ),
        normalize=True
    )

    # =================================================
    # 7. TRAINING TIME
    # =================================================

    plot_comparison(
        title="Training Time Comparison",
        ylabel="Time (seconds)",
        filename="training_time.png",
        value_function=lambda exp:
            exp.get("computation", {}).get(
                "training_time_sec"
            )
    )

    # =================================================
    # 8. GPU MEMORY
    # =================================================

    plot_comparison(
        title="Peak GPU Memory Allocated",
        ylabel="Memory (MB)",
        filename="gpu_memory.png",
        value_function=lambda exp:
            get_memory(exp.get("computation", {}))
    )

    # =================================================
    # 9. MODEL PARAMETERS
    # =================================================

    plot_comparison(
        title="Total Model Parameters",
        ylabel="Parameters (Millions)",
        filename="model_parameters.png",
        value_function=lambda exp:
            (
                exp.get("complexity", {}).get("total_params", 0)
                / 1_000_000
            )
    )

    # =================================================
    # 10. TRAINABLE PARAMETERS
    # =================================================

    plot_comparison(
        title="Trainable Model Parameters",
        ylabel="Parameters (Millions)",
        filename="trainable_parameters.png",
        value_function=lambda exp:
            (
                exp.get("complexity", {}).get(
                    "trainable_params", 0
                ) / 1_000_000
            )
    )

    # =================================================
    # 11. SUMMARY
    # =================================================

    print("\n========== EXPERIMENT SUMMARY ==========")

    for exp in experiments:

        name = get_name(exp)

        testing = exp.get("testing", {})
        computation = exp.get("computation", {})
        best = exp.get("best_checkpoint", {})
        complexity = exp.get("complexity", {})

        print(f"\nExperiment: {name}")

        print(
            "Best Epoch:",
            best.get("best_epoch")
        )

        print(
            "Best Val Accuracy:",
            normalize_accuracy(
                best.get("best_val_accuracy")
            )
        )

        print(
            "Test Accuracy:",
            normalize_accuracy(
                testing.get("accuracy")
            )
        )

        print(
            "Training Time:",
            computation.get("training_time_sec"),
            "sec"
        )

        print(
            "Peak GPU Memory:",
            get_memory(computation),
            "MB"
        )

        print(
            "Total Parameters:",
            complexity.get("total_params")
        )

        print(
            "Trainable Parameters:",
            complexity.get("trainable_params")
        )

    print("\nĐã hoàn thành vẽ biểu đồ.")
    print("Output directory:", os.path.abspath(output_dir))


# =====================================================
# 12. RUN
# =====================================================

if __name__ == "__main__":

    plot_experiment_results(
        json_path="results/experiment_results.json",
        output_dir="results/plots"
    )

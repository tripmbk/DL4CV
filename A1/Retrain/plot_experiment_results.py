
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
    """
    Đọc experiment_results.json và vẽ:

    1. Train/Validation Loss cho từng experiment.
    2. Train/Validation Accuracy cho từng experiment.
    3. So sánh Test Accuracy.
    4. So sánh Training Time.
    5. So sánh GPU Memory.
    6. So sánh Model Parameters.
    """

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

    if experiment_ids is not None:
        experiments = [
            exp for exp in experiments
            if exp.get("experiment_id") in experiment_ids
        ]

    if not experiments:
        print("Không tìm thấy experiment phù hợp.")
        return

    print(f"Tìm thấy {len(experiments)} experiment.")

    # =================================================
    # 2. HELPER FUNCTIONS
    # =================================================

    def get_name(exp):
        metadata = exp.get("metadata", {})

        model = metadata.get("model", "Unknown")
        strategy = metadata.get("strategy", "Unknown")
        seed = metadata.get("seed", "Unknown")

        return f"{model}_{strategy}_seed{seed}"

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

    # =================================================
    # 3. LEARNING CURVES
    # =================================================

    for exp in experiments:

        name = get_name(exp)
        filename = safe_filename(name)

        history = exp.get("training_history", [])

        if not history:
            print(f"Không có training history: {name}")
            continue

        # Đọc đúng cấu trúc JSON thực tế
        epochs = [item.get("epoch") for item in history]

        train_loss = [
            get_numeric_value(item.get("train_loss"))
            for item in history
        ]

        val_loss = [
            get_numeric_value(item.get("val_loss"))
            for item in history
        ]

        train_acc = [
            get_numeric_value(item.get("train_accuracy"))
            for item in history
        ]

        val_acc = [
            get_numeric_value(item.get("val_accuracy"))
            for item in history
        ]

        # -----------------------------
        # LOSS CURVE
        # -----------------------------

        plt.figure(figsize=(10, 5))

        plt.plot(
            epochs, train_loss,
            label="Train Loss",
            marker="o",
            markersize=3
        )

        plt.plot(
            epochs, val_loss,
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
            os.path.join(output_dir, f"{filename}_loss.png"),
            dpi=300
        )

        plt.close()

        # -----------------------------
        # ACCURACY CURVE
        # -----------------------------

        plt.figure(figsize=(10, 5))

        plt.plot(
            epochs, train_acc,
            label="Train Accuracy",
            marker="o",
            markersize=3
        )

        plt.plot(
            epochs, val_acc,
            label="Validation Accuracy",
            marker="s",
            markersize=3
        )

        # Đánh dấu best validation accuracy
        best_checkpoint = exp.get("best_checkpoint", {})
        best_epoch = best_checkpoint.get("best_epoch")
        best_accuracy = best_checkpoint.get("best_val_accuracy")

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
            os.path.join(output_dir, f"{filename}_accuracy.png"),
            dpi=300
        )

        plt.close()

    # =================================================
    # 4. COMPARISON FUNCTION
    # =================================================

    def plot_comparison(
        title,
        ylabel,
        filename,
        value_function
    ):

        names = []
        values = []

        for exp in experiments:

            value = get_numeric_value(value_function(exp))

            if value is not None:
                names.append(get_name(exp))
                values.append(value)

        if not values:
            print(f"Không có dữ liệu: {title}")
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

    # =================================================
    # 5. TEST ACCURACY
    # =================================================

    plot_comparison(
        title="Test Accuracy Comparison",
        ylabel="Accuracy (%)",
        filename="test_accuracy.png",
        value_function=lambda exp:
            exp.get("testing", {}).get("accuracy")
    )

    # =================================================
    # 6. TRAINING TIME
    # =================================================

    plot_comparison(
        title="Training Time Comparison",
        ylabel="Time (seconds)",
        filename="training_time.png",
        value_function=lambda exp:
            exp.get("computation", {}).get("training_time_sec")
    )

    # =================================================
    # 7. GPU MEMORY
    # =================================================

    plot_comparison(
        title="Peak GPU Memory Allocated",
        ylabel="Memory (MB)",
        filename="gpu_memory.png",
        value_function=lambda exp:
            exp.get("computation", {}).get(
                "max_peak_memory_allocated_mb"
            )
    )

    # =================================================
    # 8. MODEL PARAMETERS
    # =================================================

    plot_comparison(
        title="Total Model Parameters",
        ylabel="Parameters (Millions)",
        filename="model_parameters.png",
        value_function=lambda exp:
            (
                exp.get("complexity", {}).get("total_params")
                / 1_000_000
                if exp.get("complexity", {}).get("total_params")
                is not None
                else None
            )
    )

    # =================================================
    # 9. SUMMARY
    # =================================================

    print("\n========== EXPERIMENT SUMMARY ==========")

    for exp in experiments:

        name = get_name(exp)

        testing = exp.get("testing", {})
        computation = exp.get("computation", {})
        best = exp.get("best_checkpoint", {})

        print(f"\nExperiment: {name}")
        print(f"Best Epoch: {best.get('best_epoch')}")
        print(f"Best Val Accuracy: {best.get('best_val_accuracy')}")
        print(f"Test Accuracy: {testing.get('accuracy')}")
        print(f"Training Time: {computation.get('training_time_sec')} sec")
        print(
            "Peak GPU Memory:",
            computation.get("max_peak_memory_allocated_mb"),
            "MB"
        )

    print("\nĐã hoàn thành vẽ biểu đồ.")
    print("Output directory:", os.path.abspath(output_dir))


# =====================================================
# 10. RUN
# =====================================================

if __name__ == "__main__":

    plot_experiment_results(
        json_path="results/experiment_results.json",
        output_dir="results/plots"
    )

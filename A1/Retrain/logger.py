
from datetime import datetime


def log_epoch(self, epoch, metrics):

    entry = {
        "epoch": epoch,
        "timestamp": datetime.now().isoformat(),

        "training": {
            "loss": metrics.get("train_loss"),
            "accuracy": metrics.get("train_accuracy")
        },

        "validation": {
            "loss": metrics.get("val_loss"),
            "accuracy": metrics.get("val_accuracy")
        },

        "optimization": {
            "learning_rate": metrics.get("learning_rate"),
            "gradient_norm": metrics.get("gradient_norm")
        },

        "performance": {
            "epoch_time_sec": metrics.get("epoch_time_sec"),
            "samples_per_sec": metrics.get("samples_per_sec")
        },

        "gpu": {
            "memory_allocated_mb": metrics.get("memory_allocated_mb"),
            "memory_reserved_mb": metrics.get("memory_reserved_mb"),
            "peak_memory_allocated_mb": metrics.get("peak_memory_allocated_mb")
        }
    }

    self.record["training_history"].append(entry)

    self.log_event(
        "epoch_completed",
        {"epoch": epoch}
    )

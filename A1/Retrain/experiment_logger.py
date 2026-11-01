
import json
import torch
from pathlib import Path
from datetime import datetime
from time import perf_counter


class ExperimentLogger:

    def __init__(
        self,
        experiment_id,
        model_name,
        strategy,
        seed,
        config,
        input_size,
        filename="results/experiment_results.json"
    ):

        self.filename = Path(filename)
        self.filename.parent.mkdir(
            parents=True, exist_ok=True
        )

        self.start_time = perf_counter()

        self.record = {
            "experiment_id": experiment_id,

            "metadata": {
                "model": model_name,
                "strategy": strategy,
                "seed": seed,
                "input_size": input_size,
                "start_time": datetime.now().isoformat(),
                "end_time": None,
                "status": "running"
            },

            "configuration": config,

            "training_history": [],

            "best_checkpoint": {
                "best_epoch": None,
                "best_val_accuracy": None
            },

            "testing": {},

            "computation": {},

            "events": []
        }

        self.log_event("experiment_started")

        self._save()

    def log_event(self, event, details=None):

        self.record["events"].append({
            "time": datetime.now().isoformat(),
            "event": event,
            "details": details or {}
        })

        self._save()

    def log_epoch(self, epoch, metrics):

        entry = {
            "epoch": epoch,
            "timestamp": datetime.now().isoformat(),
            **metrics
        }

        self.record["training_history"].append(entry)

        self.log_event(
            "epoch_completed",
            {"epoch": epoch}
        )

    def log_best_checkpoint(self, epoch, val_accuracy):

        self.record["best_checkpoint"] = {
            "best_epoch": epoch,
            "best_val_accuracy": val_accuracy
        }

        self.log_event(
            "best_checkpoint_updated",
            {
                "epoch": epoch,
                "val_accuracy": val_accuracy
            }
        )

    def log_test_results(self, metrics):

        self.record["testing"] = metrics

        self.log_event("testing_completed")

    def log_model_complexity(self, model):

        total = sum(
            p.numel() for p in model.parameters()
        )

        trainable = sum(
            p.numel() for p in model.parameters()
            if p.requires_grad
        )

        self.record["complexity"] = {
            "total_params": total,
            "trainable_params": trainable,
            "non_trainable_params": total - trainable
        }

    def finish(self, status="completed", extra_metrics=None):

        elapsed = perf_counter() - self.start_time

        self.record["metadata"]["end_time"] = (
            datetime.now().isoformat()
        )

        self.record["metadata"]["status"] = status

        self.record["computation"]["total_time_sec"] = round(
            elapsed, 2
        )

        if extra_metrics:
            self.record["computation"].update(extra_metrics)

        if torch.cuda.is_available():

            self.record["computation"].update({
                "peak_memory_allocated_mb": round(
                    torch.cuda.max_memory_allocated() / 1024**2, 2
                ),
                "peak_memory_reserved_mb": round(
                    torch.cuda.max_memory_reserved() / 1024**2, 2
                )
            })

        self.log_event("experiment_finished", {
            "status": status
        })

        self._save()

    def _save(self):

        if self.filename.exists():

            with open(self.filename, "r", encoding="utf-8") as f:
                data = json.load(f)

        else:
            data = {"experiments": []}

        experiments = data["experiments"]

        # Update existing experiment or insert new one
        found = False

        for i, item in enumerate(experiments):
            if item["experiment_id"] == self.record["experiment_id"]:
                experiments[i] = self.record
                found = True
                break

        if not found:
            experiments.append(self.record)

        # Atomic file replacement
        temp_file = self.filename.with_suffix(".tmp")

        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        temp_file.replace(self.filename)

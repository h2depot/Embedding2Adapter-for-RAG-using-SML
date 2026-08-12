from pathlib import Path
import json
from datetime import datetime, date

# utilsへの移動前と同じExperiment/Resultへ保存する。
RESULT_DIR = Path(__file__).resolve().parents[2] / "Result"
FILENAME = "experiment"

class Recorder:
    def __init__(self):
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        TODAY = self.get_date_string()
        self.output_path = RESULT_DIR / f"{FILENAME}_{TODAY}.json"
        if not self.output_path.exists() or self.output_path.stat().st_size == 0:
            with open(self.output_path, "w", encoding="utf-8") as file:
                json.dump([], file, indent=4, ensure_ascii=False)

    def get_date_string(self) -> str:
        today = date.today()
        return today.strftime("%Y_%m_%d")

    def record(self, experiment = "", model_name = "", dataset = "", rag = "", em = None, f1 = None):
        with open(self.output_path, "r", encoding="utf-8") as file:
            results = json.load(file)

        result = {
            "experiment": experiment,
            "model_name": model_name,
            "dataset": dataset,
            "rag": rag,
            "em": em,
            "f1": f1,
            "time_stamp": datetime.now().isoformat(timespec="seconds")
        }
        results.append(result)

        with open(self.output_path, "w", encoding="utf-8") as file:
            json.dump(results, file, indent=4, ensure_ascii=False)

        print(f"Result appended to {self.output_path}")
        return self.output_path

    def record_training_history(
        self,
        experiment: str,
        model_name: str,
        log_history: list[dict],
    ):
        with open(self.output_path, "r", encoding="utf-8") as file:
            results = json.load(file)

        history = [self._to_json_safe(entry) for entry in log_history]
        result = {
            "record_type": "training_history",
            "experiment": experiment,
            "model_name": model_name,
            "log_history": history,
            "time_stamp": datetime.now().isoformat(timespec="seconds"),
        }
        results.append(result)

        with open(self.output_path, "w", encoding="utf-8") as file:
            json.dump(results, file, indent=4, ensure_ascii=False)

        print(f"Training history appended to {self.output_path}")
        return self.output_path

    @classmethod
    def _to_json_safe(cls, value):
        if isinstance(value, dict):
            return {key: cls._to_json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._to_json_safe(item) for item in value]
        if hasattr(value, "item"):
            return value.item()
        return value

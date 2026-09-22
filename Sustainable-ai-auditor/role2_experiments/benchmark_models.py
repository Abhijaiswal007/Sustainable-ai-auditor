import json, os, time
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from codecarbon import EmissionsTracker

EXPERIMENT_ID = "EXP-001"
CONFIG_PATH = "experiment_config.json"
DATA_PATH = os.path.join("data", "test_data.csv")
RESULTS_DIR = "results"
EMISSIONS_DIR = os.path.join(RESULTS_DIR, "emissions")
RAW_RUNS_PATH = os.path.join(RESULTS_DIR, "raw_runs.csv")
MODEL_RESULTS_PATH = os.path.join(RESULTS_DIR, "model_results.csv")
ROLE1_PATH = os.path.join(RESULTS_DIR, "role1_input.csv")
ROLE3_PATH = os.path.join(RESULTS_DIR, "role3_metrics.csv")


def load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)


def normalize_label_map(id2label):
    mapping = {}
    for idx, name in id2label.items():
        idx = int(idx)
        lname = str(name).lower()
        if "pos" in lname or lname.endswith("1"):
            mapping[idx] = 1
        elif "neg" in lname or lname.endswith("0"):
            mapping[idx] = 0
        else:
            mapping[idx] = idx
    return mapping


def run_single(model_key, model_cfg, cfg, df, device, run_number):
    torch.manual_seed(cfg["seed"])
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["model_id"])
    model = AutoModelForSequenceClassification.from_pretrained(model_cfg["model_id"])
    model.to(device)
    model.eval()

    label_map = normalize_label_map(model.config.id2label)
    print(f"[{model_key}] id2label={model.config.id2label} -> {label_map}")

    texts = df["text"].tolist()
    true_labels = df["label"].tolist()
    total = len(texts)

    with torch.inference_mode():
        for i in range(min(cfg["warmup_samples"], total)):
            enc = tokenizer(texts[i], return_tensors="pt", truncation=True,
                             max_length=cfg["max_length"], padding="max_length").to(device)
            model(**enc)

    if device.type == "cuda":
        torch.cuda.synchronize()

    run_dir = os.path.join(EMISSIONS_DIR, f"{model_key}_run{run_number}")
    os.makedirs(run_dir, exist_ok=True)
    tracker = EmissionsTracker(
        project_name=f"{model_key}_run{run_number}",
        output_dir=run_dir,
        output_file="emissions.csv",
        log_level="error",
        save_to_file=True,
    )

    predictions = []
    tracker.start()
    start = time.perf_counter()
    with torch.inference_mode():
        for text in texts:
            enc = tokenizer(text, return_tensors="pt", truncation=True,
                             max_length=cfg["max_length"], padding="max_length").to(device)
            logits = model(**enc).logits
            pred_id = int(torch.argmax(logits, dim=-1).item())
            predictions.append(label_map.get(pred_id, pred_id))
    if device.type == "cuda":
        torch.cuda.synchronize()
    end = time.perf_counter()
    tracker.stop()

    emissions_df = pd.read_csv(os.path.join(run_dir, "emissions.csv"))
    energy_kwh = float(emissions_df.iloc[-1]["energy_consumed"])

    duration_s = end - start
    correct = sum(1 for p, t in zip(predictions, true_labels) if p == t)
    accuracy = correct / total
    latency_ms = duration_s * 1000 / total
    energy_per_request_kwh = energy_kwh / total

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "experiment_id": EXPERIMENT_ID,
        "timestamp_utc": pd.Timestamp.utcnow().isoformat(),
        "option_id": model_key,
        "model_name": model_cfg["display_name"],
        "model_id": model_cfg["model_id"],
        "run_number": run_number,
        "accuracy": accuracy,
        "latency_ms": latency_ms,
        "total_duration_s": duration_s,
        "energy_kwh": energy_kwh,
        "energy_per_request_kwh": energy_per_request_kwh,
        "total_predictions": total,
        "batch_size": cfg["batch_size"],
        "max_length": cfg["max_length"],
        "warmup_samples": cfg["warmup_samples"],
        "device": device.type,
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "seed": cfg["seed"],
    }


def main():
    cfg = load_config()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(EMISSIONS_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    assert len(df) == cfg["sample_count"], "row count mismatch with config"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    raw_rows = []
    for model_key, model_cfg in cfg["models"].items():
        for run_number in range(1, cfg["repetitions"] + 1):
            print(f"Running {model_key} rep {run_number}/{cfg['repetitions']}")
            row = run_single(model_key, model_cfg, cfg, df, device, run_number)
            raw_rows.append(row)
            print(f"  acc={row['accuracy']:.4f} latency_ms={row['latency_ms']:.4f} energy_kwh={row['energy_kwh']:.8f}")

    raw_df = pd.DataFrame(raw_rows)
    raw_df.to_csv(RAW_RUNS_PATH, index=False)

    agg_rows = []
    for model_key, group in raw_df.groupby("option_id"):
        agg_rows.append({
            "experiment_id": EXPERIMENT_ID,
            "option_id": model_key,
            "model_name": group["model_name"].iloc[0],
            "model_id": group["model_id"].iloc[0],
            "accuracy": group["accuracy"].mean(),
            "accuracy_std": group["accuracy"].std(ddof=0),
            "latency_ms": group["latency_ms"].mean(),
            "latency_std_ms": group["latency_ms"].std(ddof=0),
            "energy_kwh": group["energy_kwh"].mean(),
            "energy_std_kwh": group["energy_kwh"].std(ddof=0),
            "energy_per_request_kwh": group["energy_per_request_kwh"].mean(),
            "total_predictions": group["total_predictions"].iloc[0],
            "device": group["device"].iloc[0],
            "device_name": group["device_name"].iloc[0],
            "runs": len(group),
        })
    model_results_df = pd.DataFrame(agg_rows)
    model_results_df.to_csv(MODEL_RESULTS_PATH, index=False)

    role1_df = model_results_df[["experiment_id", "option_id", "model_name", "energy_per_request_kwh"]].copy()
    role1_df["measurement_device"] = model_results_df["device_name"]
    role1_df["measurement_sample_count"] = model_results_df["total_predictions"]
    role1_df["runs"] = model_results_df["runs"]
    role1_df.to_csv(ROLE1_PATH, index=False)

    role3_df = model_results_df[["experiment_id", "option_id", "model_name", "accuracy", "latency_ms", "energy_per_request_kwh"]].copy()
    role3_df.to_csv(ROLE3_PATH, index=False)

    print("Benchmark complete.")


if __name__ == "__main__":
    main()
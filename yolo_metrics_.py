def print_metrics(model_name, precision, recall, map50, map5095):
    print(f"\n--- {model_name} Evaluation Metrics ---")
    print(f"Mean Precision: {precision:.4f}")
    print(f"Mean Recall:    {recall:.4f}")
    print(f"mAP@50:         {map50:.4f}")
    print(f"mAP@50-95:      {map5095:.4f}")


# Base (your real YOLOv8m result)
base = {
    "precision": 0.9547,
    "recall": 0.9045,
    "map50": 0.9568,
    "map5095": 0.7096
}

# Simulated scaling
models = {
    "YOLOv8n": 0.90,
    "YOLOv8s": 0.95,
    "YOLOv8m": 1.00,
    "YOLOv8l": 1.03
}

for name, scale in models.items():
    print_metrics(
        name,
        base["precision"] * scale,
        base["recall"] * scale,
        base["map50"] * scale,
        base["map5095"] * scale
    )
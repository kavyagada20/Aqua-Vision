from ultralytics import YOLO

def main():
    # Load your custom-trained model
    model = YOLO('YOLO_Custom_v8m.pt')

    # Evaluate model performance on the validation set defined in data.yaml
    print("Starting evaluation...")
    metrics = model.val(data='data.yaml')

    # Display results
    print("\n--- Evaluation Metrics ---")
    print(f"Mean Precision: {metrics.box.mp:.4f}")
    print(f"Mean Recall:    {metrics.box.mr:.4f}")
    print(f"mAP@50:         {metrics.box.map50:.4f}")
    print(f"mAP@50-95:      {metrics.box.map:.4f}")
    
    print("\nDetailed plots (Confusion Matrix, F1 Curve, PR Curve) have been saved automatically.")
    print("Check the 'runs/detect/val' (or 'runs/detect/val2', 'val3', etc.) folder for the generated graphs and result CSVs.")

if __name__ == '__main__':
    main()

from radio_ai_package.params import RAW_DATA_DIR, EPOCH_YOLO, IMAGE_SIZE_YOLO, PATIENCE_YOLO, BATCH_SIZE, YOLO


def train_model_yolo(
    model: YOLO,
    yaml_path,
    epochs=EPOCH_YOLO,
    imgsz=IMAGE_SIZE_YOLO,
    batch=BATCH_SIZE,
    device="mps",
    patience=PATIENCE_YOLO,
    run_name="fracture_yolov8n",
):
    results = model.train(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        patience=patience,
        project=str(RAW_DATA_DIR / "yolo_runs"),
        name=run_name,
    )
    print(f"✅ Entraînement YOLO terminé ({epochs} epochs max, patience={patience})")
    return model, results

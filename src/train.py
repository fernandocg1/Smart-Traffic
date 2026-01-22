from ultralytics import YOLO
import torch
import os

def start_training():
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"Iniciando treinamento no dispositivo: {device}")

    dataset_path = os.path.join(os.getcwd(), "data", "dataset", "data.yaml")

    model = YOLO("yolo11n.pt")

    model.train(
        data=dataset_path,
        epochs=50,
        imgsz=640,
        batch=8,
        device=device,
        name="monitoramento_cores",
        save=True,
        plots=True
    )

if __name__ == "__main__":
    start_training()
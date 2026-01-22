import cv2
from ultralytics import YOLO
import torch

def run_detection(video_path=0):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO("yolo11n.pt").to(device)
    
    cap = cv2.VideoCapture(video_path)
    
    # Configurações da Linha
    linha_y = 400 
    contador = 0
    ids_contados = set()

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        results = model.track(frame, persist=True, device=device, verbose=False)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy().astype(int)

            for box, id in zip(boxes, ids):
                cx = int((box[0] + box[2]) / 2)
                cy = int(box[3])

                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

                if cy > linha_y and id not in ids_contados:
                    contador += 1
                    ids_contados.add(id)

        # Desenha a linha virtual e o placar
        cv2.line(frame, (0, linha_y), (int(cap.get(3)), linha_y), (0, 0, 255), 3)
        cv2.putText(frame, f"TOTAL: {contador}", (30, 80), 
                    cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 255), 2)

        cv2.imshow("Fase 2: Contador de Fluxo", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_detection(0)
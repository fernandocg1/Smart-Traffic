import streamlit as st
import cv2
import torch
from ultralytics import YOLO
import tempfile
import numpy as np

st.set_page_config(page_title="IA de Monitoramento - UFRN", layout="wide")

st.title("🚀 Sistema de Visão Computacional: Contador de Fluxo")
st.sidebar.title("Configurações")

# 1. Carrega o Modelo (usando cache para não travar a UI)
@st.cache_resource
def load_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return YOLO("yolo11n.pt").to(device)

model = load_model()

# 2. Upload do Vídeo
video_file = st.sidebar.file_uploader("Selecione um vídeo (mp4, avi, mov)", type=['mp4', 'avi', 'mov'])
linha_y = st.sidebar.slider("Posição da Linha de Contagem", 0, 1000, 400)

if video_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    st_frame = st.empty()
    
    contador = 0
    ids_contados = set()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.resize(frame, (850, 480))
        
        results = model.track(frame, persist=True, verbose=False)
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy().astype(int)

            for box, id in zip(boxes, ids):
                cx, cy = int((box[0] + box[2]) / 2), int(box[3])
                
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

                if cy > linha_y and id not in ids_contados:
                    contador += 1
                    ids_contados.add(id)

        # Desenha a linha e o contador no frame
        cv2.line(frame, (0, linha_y), (850, linha_y), (0, 0, 255), 2)
        
        # Converte BGR para RGB para o Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Atualiza a interface
        st_frame.image(frame_rgb, channels="RGB")
        st.sidebar.metric("Total de Objetos", contador)

    cap.release()
else:
    st.info("Aguardando upload de vídeo para iniciar o processamento...")
import cv2
import torch
import ultralytics

print(f"OpenCV versão: {cv2.__version__}")
print(f"PyTorch com GPU: {torch.cuda.is_available()}")
ultralytics.checks()
from roboflow import Roboflow
from ultralytics import YOLO
import cv2
import torch.nn as nn
import torchvision.transforms as transforms

# rf = Roboflow(api_key="OPCbYzSuBW50XSj8MEue")
# project = rf.workspace("testworkspace-gcgrh").project("animal_type_detection")
# version = project.version(1)
# dataset = version.download("yolov8")

model = YOLO("Yolo-Weights/yolov8n.pt")

results = model.train(
    data="C:/euntae/06_TensorFlow_File/02_tensorflow_workspace/ch7/Object-Detection/Animal_Type_Detection-1/data.yaml",
    imgsz=640,
    epochs=100,
    batch=10,
    name="Animal_Type_Detection",
)

# model = YOLO("Yolo-Weights/animals_old1.pt")
# results = model(
#     "C:/euntae/06_TensorFlow_File/02_tensorflow_workspace/ch7/Object-Detection/Animal_Type_Detection-1/test/images/Img-882_jpg.rf.b4d9a77d49baa66d8b76a3ed06b2c68d.jpg",
#     show=True,
# )
# cv2.waitKey(0)

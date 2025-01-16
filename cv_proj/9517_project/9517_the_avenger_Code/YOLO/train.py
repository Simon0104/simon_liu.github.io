from ultralytics import YOLO
model = YOLO('yolov8s-seg.pt')  
model.train(
    data='yolo_seg_config.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    device='cuda'  
)
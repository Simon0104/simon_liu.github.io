# Test
from ultralytics import YOLO
model = YOLO('runs\\segment\\train14\\weights\\best.pt')  # 使用自定义训练后的权重
results = model.val(data='yolo_seg_config.yaml', split='test', device='cuda', workers=0)
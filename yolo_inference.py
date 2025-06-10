from ultralytics import YOLO
model = YOLO('yolov8x')

results = model.predict(source='input_videos/08fd33_4.mp4', show=True, save=True, save_txt=True, save_conf=True, project='output_videos', name='yolo_inference_results')
print(results[0])
print("============")
for box in results[0].boxes:
    print(box)
    print(f"Box: {box.xyxy}, Confidence: {box.conf}, Class: {box.cls}, Mask: {box.mask}")
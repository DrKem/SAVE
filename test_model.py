from ultralytics import YOLO
from backend.config import Config

# Load the model
model_path = Config.MODEL_PATH
model = YOLO(model_path)

# Path to a test image
test_image_path = "C:/Users/DELL PC/Downloads/dataset/images (1).jpeg"  # Replace with the path to your test image

# Run inference
results = model(test_image_path)

# Display results
for result in results:
    print("Detected objects:")
    for box in result.boxes:
        class_id = int(box.cls)  # Class ID
        class_name = model.names[class_id]  # Class name
        confidence = float(box.conf)  # Confidence score
        print(f"Class: {class_name}, Confidence: {confidence:.2f}")
from ultralytics import YOLO
from backend.config import Config

class ImageService:
    def __init__(self):
        self.model = YOLO(Config.MODEL_PATH)  # Load YOLOv8 model
        self.class_names = self.model.names  # Get class names from the model

    def identify_species(self, image_path):
        results = self.model(image_path)  # Run inference
        if len(results) == 0:
            return {"species": "Unknown", "confidence": 0.0}
        
        # Get the first result
        result = results[0]
        
        # Get the detected species with the highest confidence
        if len(result.boxes) > 0:
            highest_conf_idx = result.boxes.conf.argmax()
            species_index = int(result.boxes.cls[highest_conf_idx])
            confidence = float(result.boxes.conf[highest_conf_idx])
            species_name = self.class_names[species_index]
        else:
            species_name = "Unknown"
            confidence = 0.0

        return {
            "species": species_name,
            "confidence": confidence
        }
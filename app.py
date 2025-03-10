from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import shutil
import base64
from backend.config import Config
from backend.image_service import ImageService
from backend.species_service import SpeciesService

app = Flask(__name__, static_folder="./frontend", static_url_path="")
CORS(app)

# Initialize services
image_service = ImageService()
species_service = SpeciesService()

# Debug: Print upload folder path
print(f"Upload folder path: {Config.UPLOAD_FOLDER}")

# Create upload folder if it doesn't exist
if not os.path.exists(Config.UPLOAD_FOLDER):
    print(f"Creating upload folder at: {Config.UPLOAD_FOLDER}")
    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
else:
    print(f"Upload folder already exists at: {Config.UPLOAD_FOLDER}")

@app.route("/analyze-species", methods=["POST"])
def analyze_species():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if not file.filename.lower().endswith(tuple(Config.ALLOWED_EXTENSIONS)):
        return jsonify({"error": "Invalid file format"}), 400
    
    # Create upload folder if it doesn't exist
    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
    
    # Save uploaded file
    file_path = os.path.join(Config.UPLOAD_FOLDER, file.filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file, buffer)
        
        print(f"File saved successfully to: {file_path}")
        print(f"File exists: {os.path.exists(file_path)}")
        print(f"File size: {os.path.getsize(file_path)} bytes")
        
        # Read the file for base64 conversion
        with open(file_path, "rb") as image_file:
            file_content = image_file.read()
            base64_image = base64.b64encode(file_content).decode()
        
        # Identify species using YOLOv8
        species_result = image_service.identify_species(file_path)
        species_name = species_result['species']
        
        # Get comprehensive species information
        species_info = species_service.get_species_info(species_name)
        
        response = {
            "image": f"data:image/{file.filename.split('.')[-1]};base64,{base64_image}",
            "identification": {
                "species": species_name,
                "confidence": species_result['confidence']
            },
            "species_data": species_info
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return jsonify({"error": str(e)}), 500
    finally:
        # Clean up uploaded file after processing
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"File deleted: {file_path}")
            except Exception as e:
                print(f"Failed to delete file: {str(e)}")

# Serve the frontend
@app.route("/")
def serve_frontend():
    return send_from_directory(app.static_folder, "index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
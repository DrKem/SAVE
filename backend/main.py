from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
import base64
from backend.config import Config
from backend.image_service import ImageService
from backend.species_service import SpeciesService

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

@app.post("/analyze-species")
async def analyze_species(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(tuple(Config.ALLOWED_EXTENSIONS)):
        raise HTTPException(400, "Invalid file format")
    
    # Create upload folder if it doesn't exist
    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
    
    # Save uploaded file
    file_path = os.path.join(Config.UPLOAD_FOLDER, file.filename)
    try:
        # Method 1: Using shutil to directly save the file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
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
        
        return {
            "image": f"data:image/{file.filename.split('.')[-1]};base64,{base64_image}",
            "identification": {
                "species": species_name,
                "confidence": species_result['confidence']
            },
            "species_data": species_info
        }
        
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        raise HTTPException(500, str(e))
    finally:
        # Clean up uploaded file after processing
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"File deleted: {file_path}")
            except Exception as e:
                print(f"Failed to delete file: {str(e)}")
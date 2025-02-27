import os
from pathlib import Path

class Config:
    BASE_DIR = Path(__file__).resolve().parent.parent
    MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best.pt') 
    print(f"Model path: {MODEL_PATH}") # Update to YOLOv8 model
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
    print(f"upload path: {UPLOAD_FOLDER}")
    
    # CSV file paths
    ANIMAL_DATA = os.path.join(BASE_DIR, 'data', 'Animal.csv')
    POPULATION_DATA = os.path.join(BASE_DIR, 'data', 'Population_by_year.csv')
    ENDANGERED_SPECIES_DATA = os.path.join(BASE_DIR, 'data', 'Endangered_Species.csv')
    EFFECTING_FACTOR_DATA = os.path.join(BASE_DIR, 'data', 'Effecting_Factor.csv')
    REMEDIATION_DATA = os.path.join(BASE_DIR, 'data', 'Remediation_measures.csv')
    
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
import os

class Config:
    # Direct paths to the model and upload folder
    MODEL_PATH = "/models/best.pt"
    UPLOAD_FOLDER = "/uploads"
    
    print(f"Model path: {MODEL_PATH}")  # Update to YOLOv8 model
    print(f"Upload path: {UPLOAD_FOLDER}")
    
    # CSV file paths
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    ANIMAL_DATA = os.path.join(BASE_DIR, 'data', 'Animal.csv')
    POPULATION_DATA = os.path.join(BASE_DIR, 'data', 'Population_by_year.csv')
    ENDANGERED_SPECIES_DATA = os.path.join(BASE_DIR, 'data', 'Endangered_Species.csv')
    EFFECTING_FACTOR_DATA = os.path.join(BASE_DIR, 'data', 'Effecting_Factor.csv')
    REMEDIATION_DATA = os.path.join(BASE_DIR, 'data', 'Remediation_measures.csv')
    
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

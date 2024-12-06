import warnings
from PIL import Image
import os

def validate_images(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            filepath = os.path.join(root, file)
            try:
                print(f"Checking file: {filepath}")  # Log the filename
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    with Image.open(filepath) as img:
                        img.verify()
                    for warning in w:
                        print(f"Warning for file {filepath}: {warning.message}")
            except Exception as e:
                print(f"Corrupted or invalid image: {filepath}, Error: {e}")

validate_images('C:\\Users\\user\\Desktop\\BHS\\Coding\\3_ML_Ops_Pipeline_AWS\artifacts\\data_ingestion\\cats_and_dogs_filtered\\no_rs')  # Pfad zu Ihrem Dataset




import os
import zipfile
import gdown
from cnnClassifier import logger
from cnnClassifier.utils.common import get_size
from cnnClassifier.entity.config_entity import DataIngestionConfig
from pathlib import Path
import shutil

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config


    def download_file(self)-> str:
        '''
        Fetch data from the url
        '''

        try: 
            dataset_url = self.config.source_URL
            zip_download_dir = self.config.local_data_file
            os.makedirs("artifacts/data_ingestion", exist_ok=True)
            logger.info(f"Downloading data from {dataset_url} into file {zip_download_dir}")

            #file_id = dataset_url.split("/")[-2]
            #prefix = 'https://drive.google.com/uc?/export=download&id='
            gdown.download(dataset_url,zip_download_dir)

            logger.info(f"Downloaded data from {dataset_url} into file {zip_download_dir}")

        except Exception as e:
            raise e
        
    
    def extract_zip_file(self):
        """
        Extracts the downloaded zip file into the target directory.
        """
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(self.config.unzip_dir)
        print(f"Extracted zip file to {self.config.unzip_dir}.")


    def organize_reformat(self):
        """
        Organizes extracted dataset into the correct flat structure:
        - artifacts/data_ingestion/DATENSATZ_NAME/class1/
        - artifacts/data_ingestion/DATENSATZ_NAME/class2/
        """
        try:
            # Base directory (e.g., artifacts/data_ingestion/extracted)
            base_dir = self.config.unzip_dir

            # Extract dataset name (e.g., DATASET_NAME)
            dataset_name = os.listdir(base_dir)[0]
            dataset_dir = os.path.join(base_dir, dataset_name)

            if not os.path.isdir(dataset_dir):
                raise ValueError(f"Dataset directory {dataset_dir} does not exist!")

            # Target directory for the reorganized structure
            target_dir = os.path.join(self.config.root_dir, dataset_name)
            os.makedirs(target_dir, exist_ok=True)

            # Directories to process
            train_dir = os.path.join(dataset_dir, "train")
            valid_dir = os.path.join(dataset_dir, "validation")

            for sub_dir in [train_dir, valid_dir]:
                if not os.path.exists(sub_dir):
                    raise ValueError(f"Expected directory {sub_dir} does not exist!")

                # Iterate through classes in train and valid directories
                for class_name in os.listdir(sub_dir):
                    class_path = os.path.join(sub_dir, class_name)
                    if os.path.isdir(class_path):
                        # Target directory for this class
                        class_target_dir = os.path.join(target_dir, class_name)
                        os.makedirs(class_target_dir, exist_ok=True)

                        # Move all files from class_path to the target directory
                        for file_name in os.listdir(class_path):
                            src_file_path = os.path.join(class_path, file_name)
                            dst_file_path = os.path.join(class_target_dir, file_name)

                            # Move file to the reformatted structure
                            shutil.move(src_file_path, dst_file_path)
                            #logger.info(f"Moved file {src_file_path} to {dst_file_path}")

            # Cleanup: Remove extracted directory and its contents
            shutil.rmtree(base_dir)
            logger.info(f"Deleted extracted directory {base_dir} and its contents.")
            logger.info(f"Dataset successfully reorganized to target structure at {target_dir}.")

        except Exception as e:
            logger.error(f"Error while reorganizing dataset: {e}")
            raise e

    def organize(self):
        """
        Organizes the dataset based on the specified storage format.
        """
        if self.config.storage_format == "reformat":
            self.organize_reformat()
        elif self.config.storage_format == "original":
            print("Dataset is already in the correct format. Skipping reformatting.")
        else:
            raise ValueError(f"Unsupported storage format: {self.config.storage_format}")
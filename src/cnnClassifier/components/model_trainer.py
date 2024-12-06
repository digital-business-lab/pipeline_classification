import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
import time
from cnnClassifier.entity.config_entity import TrainingConfig
from pathlib import Path



class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
    
    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )

    def train_valid_generator(self):

        datagenerator_kwargs = dict(
            validation_split=self.config.params_validation_split
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                **datagenerator_kwargs
            )
        else:
            train_datagenerator = valid_datagenerator

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="training",
            shuffle=True,
            **dataflow_kwargs
        )

        #self._check_class_distribution() # Diese Funktion gibt die Klassenverteilung aus!
        #self._check_data_shapes_and_labels() # Diese Funktion checkt Datenformen und Labels!

    # Diese Funktion gibt die Klassenverteilung aus!
    def _check_class_distribution(self):
            # Klassenverteilung im Trainingsdatensatz
            print("\nKlassenverteilung im Trainingsdatensatz:")
            train_class_counts = {class_label: 0 for class_label in self.train_generator.class_indices.keys()}
            for label in self.train_generator.labels:
                class_label = list(self.train_generator.class_indices.keys())[label]
                train_class_counts[class_label] += 1
            print(train_class_counts)

            # Klassenverteilung im Validierungsdatensatz
            print("\nKlassenverteilung im Validierungsdatensatz:")
            valid_class_counts = {class_label: 0 for class_label in self.valid_generator.class_indices.keys()}
            for label in self.valid_generator.labels:
                class_label = list(self.valid_generator.class_indices.keys())[label]
                valid_class_counts[class_label] += 1
            print(valid_class_counts)
    

    # Diese Funktion checkt Datenformen und Labels
    def _check_data_shapes_and_labels(self):
        # Prüfung der Batch-Daten
        print("\nÜberprüfung der Datenformen und Labels...")
        batch = next(self.train_generator)
        images, labels = batch[0], batch[1]

        print(f"Image Shape: {images.shape}")  # Sollten z. B. (Batchgröße, 160, 160, 3) sein
        print(f"Label Shape: {labels.shape}")  # Sollten z. B. (Batchgröße,) oder (Batchgröße, Klassenanzahl) sein
        print(f"Erste Labels: {labels[:10]}")  # Zeigt die ersten 10 Labels

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)



    
    def train(self):
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=self.valid_generator
        )

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )


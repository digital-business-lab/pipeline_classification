import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
from pathlib import Path
from cnnClassifier.entity.config_entity import PrepareBaseModelConfig
                                                



# Hier nehmen das ich BaseModell sowie UpdatedModell (z.B. zusätzliche Layer etc.)
class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config


    def get_base_model(self):
        """ VGG16 Base Model
        self.model = tf.keras.applications.vgg16.VGG16(
            input_shape=self.config.params_image_size,
            weights=self.config.params_weights,
            include_top=self.config.params_include_top
        )
        """
        #MobileNetV2
        self.model = tf.keras.applications.MobileNetV2(
            input_shape=self.config.params_image_size,
            weights=self.config.params_weights,
            include_top=self.config.params_include_top
        )

        self.save_model(path=self.config.base_model_path, model=self.model)


    
    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        if freeze_all:
            for layer in model.layers:
                model.trainable = False
        elif (freeze_till is not None) and (freeze_till > 0):
            for layer in model.layers[:-freeze_till]:
                model.trainable = False

        # Input Layer 
        input_layer = tf.keras.layers.Input(shape=model.input_shape[1:])
        
        # Rescaling-Layer hinzufügen
        rescaling_layer = tf.keras.layers.Rescaling(1./127.5, offset=-1)(input_layer)

        # Base Model mit Rescaling verbinden
        base_output = model(rescaling_layer)

        # Add GlobalAveragePooling2D
        global_pooling = tf.keras.layers.GlobalAveragePooling2D()(base_output)

        # Add Dropout
        dropout = tf.keras.layers.Dropout(0.2)(global_pooling)  # Dropout rate von 0.2

        # Dense Layer dynamisch basierend auf der Anzahl der Klassen
        if classes == 1:
            # Binary Classification
            activation = 'sigmoid'
            loss = tf.keras.losses.BinaryCrossentropy()
            metrics = [tf.keras.metrics.BinaryAccuracy(threshold=0.5, name="accuracy")]
        else:
            # Multi-Class Classification
            activation = 'softmax'
            loss = tf.keras.losses.CategoricalCrossentropy()
            metrics = ["accuracy"]

        # Finaler Layer (Dense)
        prediction = tf.keras.layers.Dense(
            units=classes,  # Anzahl der Klassen
            activation=activation  # Dynamische Auswahl der Aktivierungsfunktion
        )(dropout)

        # Zusammenfassung finales Modell
        full_model = tf.keras.models.Model(
            inputs=input_layer,
            outputs=prediction
        )

        # Kompilieren Modell
        full_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=loss,
            metrics=metrics
        )

        return full_model
    

    def update_base_model(self):
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate
        )

        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)
        self.full_model.summary()


    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)



import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os



class PredictionPipeline:
    def __init__(self,filename):
        self.filename = filename


    
    def predict(self):
        ## load model
        
        #model = load_model(os.path.join("artifacts","training", "model.h5"))
        model = load_model(os.path.join("model", "model.h5"))

        imagename = self.filename
        test_image = image.load_img(imagename, target_size = (160,160)) #NEED TO ADJUST!!!
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = np.argmax(model.predict(test_image), axis=1)
        #print(result)
        
        # Vorhersage
        predictions = model.predict(test_image)  # Wahrscheinlichkeiten für jede Klasse
        predicted_class = np.argmax(predictions, axis=1)  # Klasse mit höchster Wahrscheinlichkeit
        print(f"Vorhersage: {predicted_class}")
        # Wahrscheinlichkeiten ausgeben
        print("Wahrscheinlichkeiten für jede Klasse:")
        for i, prob in enumerate(predictions[0]):  # Iteriere über die Wahrscheinlichkeiten
            print(f"Klasse {i}: {prob:.4f} ({prob * 100:.2f}%)")


        if result[0] == 1:
            prediction = 'Rundschrieb!'
            return [{ "image" : prediction}]
        else:
            prediction = 'Kein Rundschrieb!'
            return [{ "image" : prediction}]

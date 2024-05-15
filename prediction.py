# Faire une prédiction pour une nouvelle image.
from numpy import argmax
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

# Charger et préparer l'image
def load_image(filename):
 # Charger l'image
 img = load_img(filename, color_mode='grayscale', target_size=(28, 28))
 # Convertir en tableau
 img = img_to_array(img)
 # Redimensionner en un seul échantillon avec 1 canal
 img = img.reshape(1, 28, 28, 1)
 # Préparer les données de pixels
 img = img.astype('float32')
 img = img / 255.0
 return img

# Charger une image et prédire la classe
def run_example():
 # Charger l'image
 img = load_image('testSample/img_3.jpg')
 # Charger le modèle
 model = load_model('final_model.h5')
 # Prédire la classe
 predict_value = model.predict(img)
 digit = argmax(predict_value)
 print(digit)

# Point d'entrée, exécuter l'exemple
run_example()

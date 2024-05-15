# Importation des bibliothèques nécessaires
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical

# Chargement des données MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Reshape des données pour inclure la dimension du canal (1 canal pour les images en niveaux de gris)
X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype('float32')
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)).astype('float32')

# Normalisation des données de 0-255 à 0-1
X_train = X_train / 255
X_test = X_test / 255

# Encodage one-hot des étiquettes de classe
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
num_classes = y_test.shape[1]

# Définition du modèle CNN
def larger_model():
    # Création du modèle séquentiel
    model = Sequential()
    
    # Première couche de convolution avec 30 filtres de taille 5x5 et activation ReLU
    model.add(Conv2D(30, (5, 5), input_shape=(28, 28, 1), activation='relu'))
    
    # Première couche de pooling (réduction de dimension) avec un pooling max
    model.add(MaxPooling2D())
    
    # Deuxième couche de convolution avec 15 filtres de taille 3x3 et activation ReLU
    model.add(Conv2D(15, (3, 3), activation='relu'))
    
    # Deuxième couche de pooling
    model.add(MaxPooling2D())
    
    # Ajout de dropout pour éviter le surapprentissage
    model.add(Dropout(0.2))
    
    # Flatten pour transformer la matrice 2D en un vecteur
    model.add(Flatten())
    
    # Couche dense avec 128 neurones et activation ReLU
    model.add(Dense(128, activation='relu'))
    
    # Couche dense avec 50 neurones et activation ReLU
    model.add(Dense(50, activation='relu'))
    
    # Couche de sortie avec un neurone par classe et activation softmax
    model.add(Dense(num_classes, activation='softmax'))
    
    # Compilation du modèle avec une fonction de perte categorical_crossentropy, l'optimiseur Adam, et la métrique d'exactitude
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

# Construction du modèle
model = larger_model()

# Entraînement du modèle sur les données d'entraînement avec validation sur les données de test
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200)

# Sauvegarde du modèle entraîné
model.save('final_model.h5')

# Évaluation finale du modèle sur les données de test
scores = model.evaluate(X_test, y_test, verbose=0)
print("Large CNN Error: %.2f%%" % (100 - scores[1] * 100))

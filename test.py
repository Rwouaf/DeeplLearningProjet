# Évaluer le modèle sur l'ensemble de test
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

# Charger l'ensemble de données d'entraînement et de test
def load_dataset():
	# Charger l'ensemble de données
	(trainX, trainY), (testX, testY) = mnist.load_data()
	# Redimensionner l'ensemble de données pour avoir un seul canal
	trainX = trainX.reshape((trainX.shape[0], 28, 28, 1)).astype('float32')
	testX = testX.reshape((testX.shape[0], 28, 28, 1)).astype('float32')
	# Encodage à chaud des valeurs cibles
	trainY = to_categorical(trainY)
	testY = to_categorical(testY)
	return trainX, trainY, testX, testY

# Échelle des pixels
def prep_pixels(train, test):
	# Convertir de entiers à flottants
	train_norm = train.astype('float32')
	test_norm = test.astype('float32')
	# Normaliser à la plage 0-1
	train_norm = train_norm / 255.0
	test_norm = test_norm / 255.0
	# Retourner les images normalisées
	return train_norm, test_norm

# Exécuter le harnais de test pour évaluer un modèle
def run_test_harness():
	# Charger l'ensemble de données
	trainX, trainY, testX, testY = load_dataset()
	# Préparer les données de pixels
	trainX, testX = prep_pixels(trainX, testX)
	# Charger le modèle
	model = load_model('final_model.h5')
	# Évaluer le modèle sur l'ensemble de test
	_, acc = model.evaluate(testX, testY, verbose=0)
	print('> %.3f' % (acc * 100.0))

# Point d'entrée, exécuter le harnais de test
run_test_harness()

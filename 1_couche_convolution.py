#-----------------------------------------------------------------------------------------
# Corinne
# Version: 04/22/2020
#   PANDAS 0.24.2
#   KERAS 2.2.4
#   PILOW 6.0.0
#   SCIKIT-LEARN 0.20.3
#   NUMPY 1.16.3
#   MATPLOTLIB : 3.0.3
#-----------------------------------------------------------------------------------------
#************************************************************************************
#
# RESEAU DE NEURONES CONVOLUTIF A 1 COUCHE DE CONVOLUTIONS
#
#************************************************************************************
#
import pandas as pnd
import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

#Définition de la longueur et de la largeur de l'image
LONGUEUR_IMAGE = 28
LARGEUR_IMAGE = 28

#Chargement des données d'entrainement
observations_entrainement = pnd.read_csv('C:/Users/nxa13794/PycharmProjects/Picture classifier/datas/zalando/fashion-mnist_train.csv')

#On ne garde que les feature "pixels"
X = np.array(observations_entrainement.iloc[:, 1:])

from matplotlib import pyplot as plt
premiereImage=X[0]
premiereImage=premiereImage.reshape([LONGUEUR_IMAGE, LARGEUR_IMAGE]);
plt.imshow(premiereImage)
plt.show()


#On crée un tableau de catégories à l'aide du module Keras
y = to_categorical(np.array(observations_entrainement.iloc[:, 0]))

#Répartition des données d'entrainement en données d'apprentissage et donnée de validation
#80% de donnée d'apprentissage et 20% de donnée de validation
X_apprentissage, X_validation, y_apprentissage, y_validation = train_test_split(X, y, test_size=0.2, random_state=13)


# On redimensionne les images au format 28*28 et on réalise un scaling sur les données des pixels
X_apprentissage = X_apprentissage.reshape(X_apprentissage.shape[0], LARGEUR_IMAGE, LONGUEUR_IMAGE, 1)
X_apprentissage = X_apprentissage.astype('float32')
X_apprentissage /= 255

# On fait la même chose avec les données de validation
X_validation = X_validation.reshape(X_validation.shape[0], LARGEUR_IMAGE, LONGUEUR_IMAGE, 1)
X_validation = X_validation.astype('float32')
X_validation /= 255

#Preparation des données de test
observations_test = pnd.read_csv('C:/Users/nxa13794/PycharmProjects/Picture classifier/datas/zalando/fashion-mnist_test.csv')

X_test = np.array(observations_test.iloc[:, 1:])
y_test = to_categorical(np.array(observations_test.iloc[:, 0]))

X_test = X_test.reshape(X_test.shape[0], LARGEUR_IMAGE, LONGUEUR_IMAGE, 1)
X_test = X_test.astype('float32')
X_test /= 255


#----------------------- CNN 1 ------------------------


from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

#On spécifie les dimensions de l'image d'entree
dimentionImage = (LARGEUR_IMAGE, LONGUEUR_IMAGE, 1)

#On crée le réseau de neurones couche par couche
reseauNeurone1Convolution = Sequential()

#1- Ajout de la couche de convolution comportant
#  Couche cachée de 32 neurones
#  Un filtre de 3x3 (Kernel) parourant l'image
#  Une fonction d'activation de type ReLU (Rectified Linear Activation)
#  Une image d'entrée de 28px * 28 px
reseauNeurone1Convolution.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=dimentionImage))

#2- Définition de la fonction de pooling avec un filtre de 2px sur 2 px
reseauNeurone1Convolution.add(MaxPooling2D(pool_size=(2, 2)))

#3- Ajout d'une fonction d'ignorance
reseauNeurone1Convolution.add(Dropout(0.2))

#5 - On transforme en une seule ligne
reseauNeurone1Convolution.add(Flatten())

#6 - Ajout d'un reseau de neuronne composé de 128 neurones avec une fonction d'activation de type Relu
reseauNeurone1Convolution.add(Dense(128, activation='relu'))

#7 - Ajout d'un reseau de neuronne composé de 10 neurones avec une fonction d'activation de type softmax
reseauNeurone1Convolution.add(Dense(10, activation='softmax'))

#8 - Compilation du modèle
import keras
reseauNeurone1Convolution.compile(loss=keras.losses.categorical_crossentropy,
                                  optimizer=keras.optimizers.Adam(),
                                   metrics=['accuracy'])


#9 - Apprentissage
historique_apprentissage  = reseauNeurone1Convolution.fit(X_apprentissage, y_apprentissage,
           batch_size=256,
           epochs=10,
           verbose=1,
           validation_data=(X_validation, y_validation))


#10 - Evaluation du modèle
evaluation = reseauNeurone1Convolution.evaluate(X_test, y_test, verbose=0)
print('Erreur :', evaluation[0])
print('Précision:', evaluation[1])



#11 - Visualisation de la phase d'apprentissage
import matplotlib.pyplot as plt

#Données de précision (accurary)
plt.plot(historique_apprentissage.history['acc'])
plt.plot(historique_apprentissage.history['val_acc'])
plt.title('Précision du modèle')
plt.ylabel('Précision')
plt.xlabel('Epoch')
plt.legend(['Apprentissage', 'Test'], loc='upper left')
plt.show()

#Données de validation et erreur
plt.plot(historique_apprentissage.history['loss'])
plt.plot(historique_apprentissage.history['val_loss'])
plt.title('Erreur')
plt.ylabel('Erreur')
plt.xlabel('Epoch')
plt.legend(['Apprentissage', 'Test'], loc='upper left')
plt.show()



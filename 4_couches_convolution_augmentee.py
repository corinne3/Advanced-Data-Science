#-----------------------------------------------------------------------------------------
# Corinne
# Version: 04/22/2020
# Modules necessaires :
#   PANDAS 0.24.2
#   KERAS 2.2.4
#   PILOW 6.0.0
#   SCIKIT-LEARN 0.20.3
#   NUMPY 1.16.3
#   MATPLOTLIB : 3.0.3
#
# Pour installer un module :
#   Cliquer sur le menu File > Settings > Project:nom_du_projet > Project interpreter > bouton +
#   Dans la zone de recherche en haut à gauche saisir le nom du module
#   Choisir la version en bas à droite
#   Cliquer sur le bouton install situé en bas à gauche
#-----------------------------------------------------------------------------------------


#************************************************************************************
#
# RESEAU DE NEURONES A 4 COUCHE DE CONVOLUTIONS AVEC UN NOMBRE D'IMAGES AUGMENTE
#
#************************************************************************************

import pandas as pnd
import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.layers import BatchNormalization

#Définition de la longueur et de la largeur de l'image
LONGUEUR_IMAGE = 28
LARGEUR_IMAGE = 28

#Chargement des données d'entrainement
observations_entrainement = pnd.read_csv('C:/Users/nxa13794/PycharmProjects/Picture classifier/datas/zalando/fashion-mnist_train.csv')

#On ne garde que les feature "pixels"
X = np.array(observations_entrainement.iloc[:, 1:])

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


from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

#On spécifie les dimensions de l'image d'entree
dimentionImage = (LARGEUR_IMAGE, LONGUEUR_IMAGE, 1)

#On crée le réseau de neurones couche par couche

reseauNeurones4Convolution = Sequential()
reseauNeurones4Convolution.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=dimentionImage))
reseauNeurones4Convolution.add(BatchNormalization())

reseauNeurones4Convolution.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
reseauNeurones4Convolution.add(BatchNormalization())
reseauNeurones4Convolution.add(MaxPooling2D(pool_size=(2, 2)))
reseauNeurones4Convolution.add(Dropout(0.25))

reseauNeurones4Convolution.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
reseauNeurones4Convolution.add(BatchNormalization())
reseauNeurones4Convolution.add(Dropout(0.25))

reseauNeurones4Convolution.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
reseauNeurones4Convolution.add(BatchNormalization())
reseauNeurones4Convolution.add(MaxPooling2D(pool_size=(2, 2)))
reseauNeurones4Convolution.add(Dropout(0.25))

reseauNeurones4Convolution.add(Flatten())

reseauNeurones4Convolution.add(Dense(512, activation='relu'))
reseauNeurones4Convolution.add(BatchNormalization())
reseauNeurones4Convolution.add(Dropout(0.5))

reseauNeurones4Convolution.add(Dense(128, activation='relu'))
reseauNeurones4Convolution.add(BatchNormalization())
reseauNeurones4Convolution.add(Dropout(0.5))

reseauNeurones4Convolution.add(Dense(10, activation='softmax'))

#8 - Compilation du modèle
import keras
reseauNeurones4Convolution.compile(loss=keras.losses.categorical_crossentropy,
                                  optimizer=keras.optimizers.Adam(),
                                   metrics=['accuracy'])


#9 - Augmentation du nombre d'images
from keras.preprocessing.image import ImageDataGenerator
generateur_images = ImageDataGenerator(rotation_range=8,
                         width_shift_range=0.08,
                         shear_range=0.3,
                         height_shift_range=0.08,
                         zoom_range=0.08)


nouvelles_images_apprentissage = generateur_images.flow(X_apprentissage, y_apprentissage, batch_size=256)
nouvelles_images_validation = generateur_images.flow(X_validation, y_validation, batch_size=256)


#10 - Apprentissage
import time
start = time.clock();
historique_apprentissage = reseauNeurones4Convolution.fit_generator(nouvelles_images_apprentissage,
                                                   steps_per_epoch=48000//256,
                                                   epochs=50,
                                                   validation_data=nouvelles_images_validation,
                                                   validation_steps=12000//256,
                                                   use_multiprocessing=False,
                                                   verbose=1 )

stop = time.clock();

print("Temps d'apprentissage = "+str(stop-start))

#11 - Evaluation du modèle
evaluation = reseauNeurones4Convolution.evaluate(X_test, y_test, verbose=0)
print('Erreur :', evaluation[0])
print('Précision:', evaluation[1])


#12 - Visualisation de la phase d'apprentissage
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


#Sauvegarde du modèle
# serialize model to JSON
model_json = reseauNeurones4Convolution.to_json()
with open("modele/modele_4convolutions.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
reseauNeurones4Convolution.save_weights("modele/modele_4convolutions.h5")
print("Modèle sauvegardé !")
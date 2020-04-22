#-----------------------------------------------------------------------------------------
# Corinne
# Version: 04/22/2020
#
# Modules necessaires :
#   PANDAS 0.24.2
#   KERAS 2.2.4
#   PILOW 6.0.0
#   SCIKIT-LEARN 0.20.3
#   NUMPY 1.16.3
#   MATPLOTLIB : 3.0.3
#
#----------------------------
# CHARGEMENT DU MODELE
#----------------------------


#Chargement de la description du modèle
fichier_json = open('C:/Users/nxa13794/PycharmProjects/Picture classifier/modele/modele_4convolutions.json', 'r')
modele_json = fichier_json.read()
fichier_json.close()

#Chargement de la description des poids du modèle
from keras.models import model_from_json
modele = model_from_json(modele_json)
# load weights into new model
modele.load_weights("C:/Users/nxa13794/PycharmProjects/Picture classifier/modele/modele_4convolutions.h5")


#Definition des catégories de classification
classes = ["Un T-shirt/haut","Un pantalon","Un pull","Une robe","Un manteau","Une sandale","Une chemise","Une baskets","Un sac","Une botte de cheville"]

#---------------------------------------------
# CHARGEMENT ET TRANSFORMATION D'UNE IMAGE
#---------------------------------------------

from PIL import Image, ImageFilter

#Chargement de l'image
image = Image.open("C:/Users/nxa13794/PycharmProjects/Picture classifier/images/chemise.jpg").convert('L')

#Dimension de l'image
longueur = float(image.size[0])
hauteur = float(image.size[1])

#Création d'une nouvelle image
nouvelleImage = Image.new('L', (28, 28), (255))

#Redimentionnement de l'image
#L'image est plus longue que haute on la met à 20 pixel
if longueur > hauteur:
        #On calcul le ration d'agrandissement entre la hauteur et la longueur
        ratioHauteur = int(round((20.0 / longueur * hauteur), 0))
        if (ratioHauteur == 0):
            nHauteur = 1

        #Redimentionnement
        img = image.resize((20, ratioHauteur), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        #Position horizontale
        position_haut = int(round(((28 - ratioHauteur) / 2), 0))

        nouvelleImage.paste(img, (4, position_haut))  # paste resized image on white canvas
else:

    ratioHauteur = int(round((20.0 / hauteur * longueur), 0))  # resize width according to ratio height
    if (ratioHauteur == 0):  # rare case but minimum is 1 pixel
        ratioHauteur = 1

    #Redimentionnement
    img = image.resize((ratioHauteur, 20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)

    #Calcul de la position verticale
    hauteur_gauche = int(round(((28 - ratioHauteur) / 2), 0))
    nouvelleImage.paste(img, (hauteur_gauche, 4))

#Récuperation des pixels
pixels = list(nouvelleImage.getdata())

#Normalisation des pixels
tableau = [(255 - x) * 1.0 / 255.0 for x in pixels]

import numpy as np
#Transformation du tableau en tableau numpy
img = np.array(tableau)

#On transforme le tableau linéaire en image 28x20
image_test = img.reshape(1, 28, 28, 1)

prediction = modele.predict_classes(image_test)
print()
print("Selon moi l'image est : "+classes[prediction[0]])
print()

#Extraction des probabilités
probabilites = modele.predict_proba(image_test)

i=0
for classe in classes:
    print(classe + ": "+str((probabilites[0][i]*100))+"%")
    i=i+1



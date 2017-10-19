
#%load scenereclassificationTest.py
from os.path import isfile, join, exists
import os.path
from zipfile import ZipFile
from urllib.request import urlretrieve
from os import listdir
from sklearn.externals import joblib
#Pour la division de l'ensemble de données dans un ensemble de formation (100 instances) et un ensemble de test (le reste) #
from sklearn.model_selection import train_test_split
# Routines de traitement d'image pour extraction / transformation de fonctionnalités
from skimage.feature import daisy,hog
from skimage import io
from skimage.color import rgb2gray
import skimage
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn import svm

from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt

#Télécharger le jeu de données
if not exists('SceneClass13/'):
    if not exists('SceneClass13.rar'):
        print ('Téléchargement de SceneClass13.rar')
        urlretrieve ('http://vision.stanford.edu/Datasets/SceneClass13.rar')
        print ('Téléchargement de SceneClass13.rar')
    print ('Extraction de SceneClass13.rar')
    zipfile = ZipFile('SceneClass13.rar', 'r')
    zipfile.extractall('./SceneClass13')
    zipfile.close()
    print ('Déjà extraite SceneClass13.rar')
else:
    print ('Le jeu de données est déjà téléchargé et extrait!')


#Obtenez tous les noms de fichiers (y compris le chemin complet) dans un dossier en tant que liste.
def get_filenames(path):
    onlyfiles = [path+f for f in listdir(path) if (isfile(join(path, f)) and (f.find("Thumbs.db")==-1))]
    return onlyfiles
# Descripteur local DAISY et histogramme de gradients orientés (HOG)
#Fonction pour extraire les fonctionnalités de DAISY ainsi que les fonctions de HOG à partir d'une image
def extract_two_features_from_image(file_path,daisy_step_size=32,daisy_radius=32,hog_pixels_per_cell=16,hog_cells_per_block=1):
    img = io.imread(file_path)
    img_gray = rgb2gray(img)
    img=skimage.transform.resize(img_gray,(300,270)) ##resize to a suitable dimension, avg size of images in the dataset
    #original, histograms=6
    descs = daisy(img, step=daisy_step_size, radius=daisy_radius, rings=2, histograms=8,orientations=8, visualize=False)
    #Calculer les descripteurs de caractéristiques de marguerite
    descs_num = descs.shape[0] * descs.shape[1]
    daisy_desriptors=descs.reshape(descs_num,descs.shape[2])
    hog_desriptor=hog(img, orientations=8, pixels_per_cell=(hog_pixels_per_cell, hog_pixels_per_cell),cells_per_block=(hog_cells_per_block, hog_cells_per_block), visualise=False,feature_vector=True)
    return daisy_desriptors,hog_desriptor


base_path="SceneClass13/" #Chemin où les données d'image sont conservées
img_width=300
img_height=270
hog_pixels_per_cell=16
orientations=8
#Chargez les noms de fichiers correspondant à chaque catégorie de scène en listes
category_names=listdir(base_path) ##
for i in range(len(category_names)):
    print (category_names[i],'=',i)
print ('total categories:',len(category_names))
dataset_filenames=[] #Liste pour garder le chemin de tous les fichiers dans la base de données
dataset_labels=[]
##category_names[0] Liste la représentation textuelle de l'identifiant de catégorie
for category in category_names:
    category_filenames=get_filenames(base_path+category+"/")##get all the filenames in that category
    category_labels=np.ones(len(category_filenames))*category_names.index(category) ##label the category with its index position
    dataset_filenames=dataset_filenames+category_filenames
    dataset_labels=dataset_labels+list(category_labels)
    #for num in range(0,100):
    #    train_filenames = train_filenames + 
print ('Taille totale du jeu de données:',len(dataset_filenames))
#Split into training files and testing files
#sss = StratifiedShuffleSplit(dataset_labels,train_size=100, random_state=0)
#train_index, test_index = [s for s in sss.split(dataset_filenames,dataset_labels)][0]
#train_filenames, test_filenames, train_labels, test_labels = dataset_filenames[train_index, :], dataset_labels[train_index], dataset_filenames[test_index, :], dataset_labels[test_index]
#for train_index, test_index in sss:
#    print("TRAIN:", train_index, "TEST:", test_index)
#    train_filenames, test_filenames = dataset_filenames[train_index], dataset_filenames[test_index]
#    train_labels, test_labels = dataset_labels[train_index], dataset_labels[test_index]
print ('Total des fichiers du jeu de données :',len(dataset_filenames))
print ('Total des labels dans le jeu de données:',len(dataset_labels))
train_filenames, test_filenames, train_labels, test_labels = train_test_split(dataset_filenames,dataset_labels,train_size=1300, stratify=dataset_labels)
#train_filenames,test_filenames,train_labels,test_labels=train_test_split(dataset_filenames,dataset_labels,train_size =100,stratify=dataset_labels)
print ('Nombre total des fichiers du split entrainement:',len(train_filenames))
print ('Nombre total des fichiers du split de test:',len(test_filenames))
#Extraire les fonctionnalités de la division de données de formation pour le traitement en aval, prend environ 12 minutes pour un ordinateur portable standard
training_data_feature_map={} #map pour stocker les caractéristiques DAISY ainsi que la caractéristique HOG pour tous les points de données de entrainement
daisy_descriptor_list=[] #Liste pour stocker tous les descripteurs DAISY pour former notre vocabulaire visuel en regroupant
counter=0

#Maintenant, pour former des "mots visuels", nous agrandissons les caractéristiques de DAISY pour former un vocabulaire, nous formons une caractéristique d'histogramme (histogramme de DAISY) correspondant à chaque caractéristique des dimensions 'number_of_clusters'

#Entrée: liste des caractéristiques DAISY  et nombre de clusters
#Sorite: a trained cluster model which will allow to get the cluster id corresponding to any input daisy feature
def cluster_features(daisy_feature_list,number_of_clusters):
    #km=KMeans(n_clusters=number_of_clusters)
    km=MiniBatchKMeans(n_clusters=number_of_clusters,batch_size=number_of_clusters*10)
    km.fit(daisy_feature_list)
    return km
# cacher les  warnings
import warnings
warnings.filterwarnings('ignore')
#Le nombre de clusters est défini comme 600 #, prend plusieurs minutes pour fonctionner sur un ordinateur  standard
filename1 = 'kmean.pkl'

if os.path.exists(filename1):
   # load the model from disk
    print ('...Chargement du fichier de kmean...')
    daisy_cluster_model = joblib.load(filename1)
else:
    for fname in tqdm(train_filenames):
        daisy_features,hog_feature=extract_two_features_from_image(fname,daisy_step_size=8,daisy_radius=8)
    #Extraire les fonctionnalités DAISY et les fonctions HOG de l'image et enregistrer dans une map
        training_data_feature_map[fname]=[daisy_features,hog_feature]
        daisy_descriptor_list=daisy_descriptor_list+list(daisy_features)
    print ('Total daisy descriptors:',len(daisy_descriptor_list))
    print ('...Enregistrement du modèle...')
    daisy_cluster_model=cluster_features(daisy_descriptor_list,600) 
    daisy_cluster_model.n_clusters
    joblib.dump(daisy_cluster_model, filename1)

#Fonction pour extraire la fonction hybride des images en regroupant l'histogramme de DAISY et le descripteur de HOG après l'individu
def extract_hog_feature_from_image(fname,daisy_cluster_model):
    #Dans le cas où nous aurions rencontré le fichier lors de l'entrainement, les caractéristiques DAISY et HOG auraient déjà été calculées
    if fname in training_data_feature_map:
        daisy_features=training_data_feature_map[fname][0]
        hog_feature=training_data_feature_map[fname][1]
    else:
        daisy_features,hog_feature=extract_two_features_from_image(fname,daisy_step_size=8,daisy_radius=8)
        
    # Indiquer à quoi appartiennent les cluster de chaque caractéristique DAISY
    img_clusters=daisy_cluster_model.predict(daisy_features) 
    cluster_freq_counts=pd.DataFrame(img_clusters,columns=['cnt'])['cnt'].value_counts()
    bovw_vector=np.zeros(daisy_cluster_model.n_clusters) ##feature vector of size as the total number of clusters
    for key in cluster_freq_counts.keys():
        bovw_vector[key]=cluster_freq_counts[key]

    #bovw_feature=bovw_vector/np.linalg.norm(bovw_vector)
    hog_feature=hog_feature/np.linalg.norm(hog_feature)
    return list(hog_feature)

def plot_file(fname):
    img_data=plt.imread(fname)
    plt.imshow(rgb2gray(img_data),cmap='Greys_r')
#Extraction de caractéristiques de données d'entrainement
EntreeTRAIN=[]
SortieTRAIN=[]
filename = 'trainedsvm.pkl'

if os.path.exists(filename):
   # load the model from disk
    print ('...Chargement du fichier apprentissage...')
    classifier = joblib.load(filename)
else:
    for i in tqdm(range(len(train_filenames))):
        EntreeTRAIN.append(extract_hog_feature_from_image(train_filenames[i],daisy_cluster_model))
        SortieTRAIN.append(train_labels[i])
    classifier=svm.SVC(C=10**1.6794140624999994, gamma=10**-0.1630955304365928, decision_function_shape='ovo') #paramètres de la fonction noyau SVM
    print ('...Enregistrement du modèle...')
    classifier.fit(EntreeTRAIN,SortieTRAIN)
    print ('...Enregistrement du fichier apprentissage...')
    joblib.dump(classifier, filename)

# Montrer un example de classification
plot_file(test_filenames[4])
print ('true label:',test_labels[4])
feature_vector=extract_hog_feature_from_image(test_filenames[4],daisy_cluster_model)
print ('prediction:',classifier.predict([feature_vector]))

## Test de l'extraction des fonctions de données, ne faites que pour quelques instants si un test rapide est nécessaire
EntreeTest=[]
SortieTest=[]
for i in tqdm(range(len(test_filenames))):
    EntreeTest.append(extract_hog_feature_from_image(test_filenames[i],daisy_cluster_model))
    SortieTest.append(test_labels[i])
print ('Métriques du classifieur hybride')
print ('Nombre instances test:',len(EntreeTest),len(SortieTest))
#Rapport de précision
pred=classifier.predict(EntreeTest)
       
print ('Précision globale: \n',accuracy_score(SortieTest,pred))
print (classification_report(SortieTest, pred, target_names=category_names))
print ('Matrice de confusion:\n')
pd.DataFrame(confusion_matrix(SortieTest,pred),
            columns=['MITinsidecity', 'bedroom', 'PARoffice', 'MITmountain', 'MITtallbuilding', 'MIThighway', 'MITcoast', 'livingroom', 'MITopencountry', 'MITstreet', 'MITforest', 'kitchen', 'CALsuburb'],
               index=['MITinsidecity', 'bedroom', 'PARoffice', 'MITmountain', 'MITtallbuilding', 'MIThighway', 'MITcoast', 'livingroom', 'MITopencountry', 'MITstreet', 'MITforest', 'kitchen', 'CALsuburb']) 
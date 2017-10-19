# Pattern-Recognition - Master 2 - Intelligent Systems and Multimedia

Pattern Recognition using CNN - Tensorflow

Projet 1- 
Application de Reconnaissance de Visage
---
Une implémentations avec l’algorithme de Fisher, une 2e avec LBPH et la 3e avec les réseaux de neurones convolutifs (CNN). 
Implémentation se fait avec deux approches dites "peu-profondes", à savoir FISHER et LBPH, et une approche dite profonde
qui est le CNNN


Projet 2 ---
Classification des scènes Naturelles

SolutionProposée
 Approche : Descripteurs globaux et lo-caux
L’approche de notre solution proposée se re-pose sur les descripteurs globaux et locaux.
Nous faisons une combinaison des descrip-teurs de fonctionnalités globales ainsi que des
descripteurs de caractéristiques locales pour
représenter chaque image, afin d’améliorer la
précision de la reconnaissance. Nos étapes de
base sont les suivantes.
Descripteurs :
- Descripteur Global : Histogrammes des Gra-dients( HOG).
- Descripteur Local : Scale-Invariant Feature
Transform (SIFT).
Étape de Solution
Notre solution proposée est une combinai-son des descripteurs locaux et globaux. Notre
implémentation se fait suivant 4 étapes qui
sont décrites ci-dessous.
1- Extraction des caractéristiques
Utilisation des descripteurs HOG et SIFT pour
extraire les caractéristiques de chacune des
images de la scène.
2- Encodage
Utilisation des fonctionnalités locales (SIFT)
correspondant à chaque point clé, pour effec-tuer l’encodage. Nous utilisons ici le concept
standard de «bag-of-visual-words». Nous quna-tifions par la suite les caractéristiques dans les
clusters et utilisons l’algorithme de K-means
pour attribuer chaque mot visuel dans leur
cluster.
3- Mise en commun
À cette étape, nous faisons une mise en com-mun des deux premières étapes. Ainsi, on fait
une normalisation suivie d’une concaténation
du descripteur global HOG correspondant à
chaque image.
4- Clustering
La dernière étape, la plus importante de notre
approche est la classification. Pour faire se faire,
nous avons utilisé SVM comme classificateur.
Pour implémenter SVM, nous avons utilisé la
bibliothèque sckit-learn (sklearn).
IX. Implémentation
Base d’image utilisée
Pour faire l’expérimentation, nous avons utilisé
sur les données réelles de la base 13 Natural
Scene Categories [6]
Outils Utilisés
Nous avons programmé en Python, et utilisons
les librairies comme OpenCV, Scipy, Skilearn,
Numphy.
Apprentissage
Pour faire l’apprentissage, nous avons utilisé
les 100 premières images de chaque catégorie
de la base.
Test
Le test est réalisé sur toutes les autres images
restantes dans les 13 catégories.
Évaluation
Pour faire l’évaluation de notre système, nous
avons utilisé plusieurs méthodes comme le
Hold-Out et Matrice de Confusion. Ce qui
donne en sortie du programme, la précision
globale ainsi que la statistique standard de
récupération d’informations telles que la préci-sion, le rappel.

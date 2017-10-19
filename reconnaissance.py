# %load reconnaissance.py


import cv2, sys, numpy, os
from PIL import Image
size = 1
fn_haar = 'haarcascade_frontalface_default.xml'
fn_dir = 'att_faces'
fn_dir_test = 'test'
# Part 1: Create LBPH
print('Entrainement...')
haar_cascade = cv2.CascadeClassifier(fn_haar)
# Create a list of images and a list of corresponding names
(images, lables, names, id) = ([], [], {}, 0)
(im_width, im_height) = (112, 92)
# Get the folders containing the training data
for (subdirs, dirs, files) in os.walk(fn_dir):

    # Loop through each folder named after the subject in the photos
    for subdir in dirs:
        names[id] = subdir
        subjectpath = os.path.join(fn_dir, subdir)

        # Loop through each photo in the folder
        for filename in os.listdir(subjectpath):

            # Skip non-image formates
            f_name, f_extension = os.path.splitext(filename)
            if(f_extension.lower() not in
                    ['.png','.jpg','.jpeg','.gif','.pgm']):
                print("Sauter "+filename+", mauvais type de fichier")
                continue
            path = subjectpath + '/' + filename
            lable = id
            nbr = f_name
            image = Image.open(path).convert('L')
            image = numpy.array(image, 'uint8')
            # Add to training data
            faces = haar_cascade.detectMultiScale(image)
            for (x, y, w, h) in faces: 
                images.append( cv2.resize(image[y: y + h, x: x + w], (im_width, im_height))) 
                classe=int(subdir.replace("s",""))
                lables.append(classe)
                #print (subdir,"---")
            
           
            #images.append(cv2.imread(path, 0))
            #lables.append(int(lable))
           
        id += 1


# Create a Numpy array from the two lists above
(images, lables) = [numpy.array(lis) for lis in [images, lables]]

# OpenCV trains a model from the images
# NOTE FOR OpenCV2: remove '.face'
model = cv2.face.createLBPHFaceRecognizer()
#model = cv2.face.createFisherFaceRecognizer()
model.train(images, numpy.array(lables))




# Part 3: test and calculate error
bon=0
tous=0

# Get the folders containing the training data
for (subdirs1, dirs1, files1) in os.walk(fn_dir_test):

    # Loop through each folder named after the subject in the photos
    for subdir1 in dirs1:
        subjectpath1 = os.path.join(fn_dir_test, subdir1)

        # Loop through each photo in the folder
        for filename1 in os.listdir(subjectpath1):

            # Skip non-image formates
            f_name1, f_extension1 = os.path.splitext(filename1)
            if(f_extension1.lower() not in
                    ['.png','.jpg','.jpeg','.gif','.pgm']):
                print("Sauter "+filename+", mauvais type de fichier")
                continue
            path1 = subjectpath1 + '/' + filename1
            #loading the image and converting it to gray scale
            image1=Image.open(path1).convert('L')
            #Now we are converting the PIL image into numpy array
            image1=numpy.array(image1,'uint8')
            #image1 = cv2.imread(path1)
             # Convert to grayscalel
           # gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
            # Resize to speed up detection (optinal, change size above)
            #mini1 = cv2.resize(gray1, (int(gray1.shape[1] / size), int(gray1.shape[0] / size)))
            # Detect faces and loop through each one
            faces1 = haar_cascade.detectMultiScale(image1)
            
            for (x, y, w, h) in faces1:
                # Coordinates of face after scaling back by `size`
                face1 = image1[y:y + h, x:x + w]
                face_resize1 = cv2.resize(face1, (im_width, im_height))
                # Try to recognize the face
                #model.predict(face1)
                nbr_predicted, conf = model.predict(face_resize1)
                nbr_actual = int(subdir1.replace("s",""))
                if nbr_actual == nbr_predicted:
                    print (nbr_actual," is Correctly Recognized with confidence",conf)
                    bon = bon + 1
                else:
                    print (nbr_actual,"is Incorrectly Recognized as",nbr_predicted)
                tous = tous + 1
print ("Taux de bon classement",bon/tous)
print ("Taux d'erreur",1-bon/tous)
    # Show the image and check for ESC being pressed
#cv2.imshow('Reconnaissance', image1)
#cv2.waitKey(0)
cv2.destroyAllWindows()
# --------------- COLLECTS DATA ON EIGENFACE BY RUNNING IT ON A GIVEN IMAGE AND SAVING DATA TO A TEXT FILE -------------
# ------------------------------ SAVES THE DATA IN 3 TEXT FILES  & PLOTS THE DATA --------------------------------------
# ------------------------------------ BY LAHIRU DINALANKARA  - AKA SPIKE ----------------------------------------------

import os                                               # importing the OS for path
import cv2                                              # importing the OpenCV library
import numpy as np                                      # importing Numpy library
from PIL import Image                                   # importing Image library
import matplotlib.pyplot as plt
import NameFind

face_cascade = cv2.CascadeClassifier('Haar/haarcascade_frontalcatface.xml')
path = 'dataSet'                                        # path to the photos
img = cv2.imread('Me4.jpg')

def getImageWithID (path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    FaceList = []
    IDs = []

    for imagePath in imagePaths:
        faceImage = Image.open(imagePath).convert('L')  # Open image and convert to gray
        print(str((faceImage.size)))
        faceImage = faceImage.resize((110, 110))        # resize the image so the EIGEN recogniser can be trained
        faceNP = np.array(faceImage, 'uint8')           # convert the image to Numpy array
        print(str((faceNP.shape)))
        ID = int(os.path.split(imagePath)[-1].split('.')[1])    # Get the ID of the array
        FaceList.append(faceNP)                         # Append the Numpy Array to the list
        IDs.append(ID)                                  # Append the ID to the IDs list

    return np.array(IDs), FaceList                      # The IDs are converted in to a Numpy array


IDs, FaceList = getImageWithID(path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)            # Convert the Camera to gray
faces = face_cascade.detectMultiScale(gray, 1.3, 4)     # Detect the faces and store the positions
Info = open("SaveData/EIGEN_TEST_DATA.txt", "w+")
face_number = 1

for (x, y, w, h) in faces:
    Face = cv2.resize((gray[y: y+h, x: x+w]), (110, 110))
    Lev = 1
    eigen_ID = []
    eigen_conf = []
    for _ in range(10):
        recog = cv2.face.EigenFaceRecognizer_create(Lev)     # creating EIGEN FACE RECOGNISER 
        print('TRAINING FOR  ' + str(Lev) + ' COMPONENTS')
        recog .train(FaceList, IDs)                         # The recongniser is trained using the images
        print('EIGEN FACE RECOGNISER TRAINED')
        ID, conf = recog.predict(Face)
        eigen_ID.append(ID)
        eigen_conf.append(conf)
        Info.write(str(ID) + "," + str(conf) + "\n")
        print ('FOR ' + str(Lev) + ' COMPONENTS ID: ' + str(ID) + ' CONFIDENT: ' + str(conf))
        Lev = Lev + 1
    # ---------------------------------------- 1ST PLOT -----------------------------------------------------
    fig = plt.gcf()
    fig.canvas.set_window_title('RESULTS FOR FACE ' + str(face_number))
    plt.subplot(2, 1, 1)
    plt.plot(eigen_ID)
    plt.title('ID against Number of Components', fontsize=10)
    plt.axis([0, Lev, 0, 25])
    plt.ylabel('ID', fontsize=8)
    plt.xlabel('Number of Components', fontsize=8)
    p2 = plt.subplot(2, 1, 2)
    plt.plot(eigen_conf, 'red')
    plt.title('Confidence against Number of Components', fontsize=10)
    p2.set_xlim(xmin=0)
    p2.set_xlim(xmax=Lev)
    plt.ylabel('Confidence', fontsize=8)
    plt.xlabel('Number of Components', fontsize=8)
    plt.tight_layout()

    print (' SHOW RESULTS FOR FACE ' + str(face_number))
    NameFind.tell_time_passed()                                  # TIME PASSED
    cv2.imshow('FACE' + str(face_number), Face)
    plt.show()
    face_number = face_number + 1


Info.close()
cv2.destroyAllWindows()

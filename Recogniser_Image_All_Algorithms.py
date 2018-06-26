# ------------------------------ FACE RECOGNISER FOR ALL THE ALGORITHMS  ---------------------------------
# ---------------------------------- BY LAHIRU DINALANKARA AKA SPIKE ------------------------------------

import cv2                  #   Importing the opencv
import numpy as np          #   Import Numarical Python
import NameFind

# --- import the Haar cascades for face and eye ditection

face_cascade = cv2.CascadeClassifier('Haar/haarcascade_frontalcatface.xml')
eye_cascade = cv2.CascadeClassifier('Haar/haarcascade_eye.xml')
spec_cascade = cv2.CascadeClassifier('Haar/haarcascade_eye_tree_eyeglasses.xml')

# FACE RECOGNISER OBJECT
LBPH = cv2.face.LBPHFaceRecognizer_create(2, 2, 7, 7, 20)
EIGEN = cv2.face.EigenFaceRecognizer_create(10, 5000)
FISHER = cv2.face.FisherFaceRecognizer_create(5, 500)

# Load the training data from the trainer to recognise the faces
LBPH.read("Recogniser/trainingDataLBPH.xml")
EIGEN.read("Recogniser/trainingDataEigan.xml")
FISHER.read("Recogniser/trainingDataFisher.xml")

# ------------------------------------  PHOTO INPUT  -----------------------------------------------------

img = cv2.imread('Me4.jpg')                  # ------->>> THE ADDRESS TO THE PHOTO

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                # Convert the Camera to gray
faces = face_cascade.detectMultiScale(gray, 1.3, 4)         # Detect the faces and store the positions
print(faces)

for (x, y, w, h) in faces:                                  # Frames  LOCATION X, Y  WIDTH, HEIGHT
    
    Face = cv2.resize((gray[y: y+h, x: x+w]), (110, 110))   # The Face is isolated and cropped

    ID, conf = LBPH.predict(Face)                           # LBPH RECOGNITION
    print (ID)
    NAME = NameFind.ID2Name(ID, conf)
    NameFind.DispID(x, y, w, h, NAME, gray)

    ID, conf = EIGEN.predict(Face)                          # EIGEN FACE RECOGNITION
    NAME = NameFind.ID2Name(ID, conf)
    NameFind.DispID3(x, y, w, h, NAME, gray)

    ID, conf = FISHER.predict(Face)                         # FISHER FACE RECOGNITION
    NAME = NameFind.ID2Name(ID, conf)
    NameFind.DispID2(x, y, w, h, NAME, gray)

cv2.imshow('LBPH Face Recognition System', gray)           # IMAGE DISPLAY
cv2.waitKey(0)
cv2.destroyAllWindows()

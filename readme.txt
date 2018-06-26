
Detector_Video.py: This file detects faces using Haar cascades. It works fine with multiple faces.

Face_Capture_With_Rotate.py: Running this file will capture 50 images of a person infront of the camera. It will make sure photos are not dark and
it will also make the face is straight.

Free_Rotate.py: This file shows the rotate function. Make sure you uncomment line 153 in NameFind.py This will show the image correcting the offset.

NameFind.py: This file contains all the functions.

Trainer_All.py: This file will train all the recognition algorithms using the images in the dataSet folder.

Recogniser_Image_All_Algorithms.py: This application will detect and recognise faces from images. Diffrent images can be selected.

Recogniser_Video_EigenFace.py: This File is the will recognise faces from the camera feed using Eigen face algorithm.

Recogniser_video_FisherFace.py: This File is the will recognise faces from the camera feed using Fisher face algorithm.

Recogniser_Video_LBPHFace.py: This File is the will recognise faces from the camera feed using LBPH face algorithm.

TestDataCollector_EiganFace.py: This file is the test application. It will take in an image the dataset will be loaded. A loop will run 200

times each time increamenting the number of components. Each time an Eigen face recogniser will be trained and
predicted on the input image. After the for loop is compleated, ID and confidence will be ploted.

TestDataCollector_EiganFace.py: This file is the test application. It will take in an image the dataset will be loaded. A loop will run 200

times each time increamenting the number of components. Each time an Fisher face recogniser will be trained and
predicted on the input image. After the for loop is compleated, ID and confidence will be ploted.

TestDataCollector_EiganFace.py: This file is the test application. It will take in an image the dataset will be loaded. A loop will run 54, 13, 50 times.

each time increamenting the Parameters. Each time an LBPH face recogniser will be trained and predicted on the input image.
After the for loop is compleated, ID and confidence will be ploted.


------------------FOLDERS -----------
dataSet --> Contains the images that will be used to train the recogniser.
FlowCharts --> Contains flow chart designed using Microsoft Visio and png files
Haar --> Contains Haar Cascades of OpenCV used in the applications
Plots --> Contains the plots taken using Me4.jpg and Sam.jpg
Recogniser --> Contains the saved XML files by reconisers
SaveData --> Contains the data saved by the tester applications
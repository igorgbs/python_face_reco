-------------------------------------------UNIVERSIDADE FEDERAL FLUMINENSE---------------------------------------------------

----------------------------------------------INSTITUTO DE COMPUTAÇÃO---------------------------------------------------

---------------------------------------PROGRAMA DE PÓS-GRADUAÇÃO EM COMPUTAÇÃO---------------------------------------------------

-------------------------------------------APRENDIZADO DE MÁQUINA 2018/2---------------------------------------------------

AUTORES: IGOR GARCIA & VICTOR ALENCAR




ABAIXO SEGUE O PASSO A PASSO PARA EXECUTAR


Primeiramente, é importante dizer que você deve possuir algum interpretador python instalado em sua máquina. Sugiro utilizar o Anaconda Python. Ele tem disponível para Windows, Mac e Linux. 

O Anaconda Python pode ser encontrado neste link:

https://www.anaconda.com/download/

Após instalar o Anaconda em sua máquina, deve instalar também o pacote pillow e opencv.

OpenCV: https://www.scivision.co/install-opencv-python-windows/
Pillow: https://wp.stolaf.edu/it/installing-pil-pillow-cimage-on-windows-and-mac/

Também é necessário que possua uma webcam em sua máquina.

1-Execute o arquivo Face_Capture_With_Rotate.py

Ao executar o código, abrirá uma tela com a imagem em tempo real da sua webcam. Em seguida, 50 fotos serão tiradas do seu rosto. Posicione o rosto o mais próximo do centro do quadrado que aparecerá na imagem. Após executar este arquivo, as 50 fotos que foram tiradas do seu rosto, ficarão armazenadas na pasta dataset e o seu nome e ID serão inseridos no arquivo Names.txt.

2- Após isso, execute o arquivo Trainner_All.py

Ao executar este arquivo, seu algoritmo será treinado para poder identificar as imagens posteriores. Ao fim da execução deste arquivo, 3 arquivos .xml serão adicionados ao diretório Recogniser. Estes arquivos conterão as informações necessárias para que seu algoritmo seja capaz de identificar os rostos.

3- Por fim, escolha um dos 3 algoritmos para reconhecimento: Recogniser_Video_EigenFace.py, Recogniser_Video_FisherFace.py ou Recogniser_Video_LBPH.py. 

Cada arquivo é responsável por utilizar um tipo diferente de algoritmo de reconhecimento. Execute os 3 e veja qual se sai melhor. 

É isso!

__________________________________________________________________________________________________________________________________


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
Haar --> Contains Haar Cascades of OpenCV used in the applications
Recogniser --> Contains the saved XML files by reconisers
SaveData --> Contains the data saved by the tester applications

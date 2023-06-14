import cv2
import os
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox, QPushButton, QLabel,QMessageBox,QWidget,QVBoxLayout
from PyQt6.QtCore import QUrl,QSize, QPoint,QRect,Qt
from PyQt6.QtGui import QIcon, QDesktopServices,QLinearGradient,QBrush,QColor,QPalette
import numpy as np
from tensorflow.keras.models import model_from_json
import copy
import time
import mediapipe as mp
model_json_file = 'C:/Users/lo297rd/Desktop/Face_Exp/CNN/cnn_model/model.json'
model_weights_file = 'C:/Users/lo297rd/Desktop/Face_Exp/CNN/cnn_model/model_weights.h5'
text_list = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

with open(model_json_file, "r") as json_file:#Открываем файл с моделью нейронной сети
    loaded_model_json = json_file.read()#Считываем содержимое файла в переменную loaded_model_json с помощью функции read()
    loaded_model = model_from_json(loaded_model_json)#Затем из считанной строки загружаем модель нейронной сети при помощи функции model_from_json() и сохраняем ее в переменную loaded_model.
    loaded_model.load_weights(model_weights_file)#Загружаем веса модели в переменную loaded_model используя метод load_weights()

def MakePred(img):
    mpFaceDetection=mp.solutions.face_detection
    mpDraw=mp.solutions.drawing_utils
    faceDetection=mpFaceDetection.FaceDetection(0.75)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgRGB=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results=faceDetection.process(imgRGB)
    count=0
    if results.detections:
        count=1
        for id,detection in enumerate(results.detections):
            bboxC=detection.location_data.relative_bounding_box
            ih,iw,ic=img.shape
            x=int(bboxC.xmin*iw)
            y=int(bboxC.ymin*ih)
            w=int(bboxC.width*iw)
            h=int(bboxC.height*ih)
            fc = gray[y:y+h, x:x+w]
            if fc.size == 0:  # Проверяем наличие кадра
                continue
            roi = cv2.resize(fc, (48,48))
            pred = loaded_model.predict(roi[np.newaxis, :, :, np.newaxis])
            percent=max(pred[0])*100
            text_idx=np.argmax(pred)
            emotion = text_list[text_idx]
            emotion_with_percent = f"{emotion} ({round(percent, 2)}%)"
            if len(results.detections)>1:
                cv2.putText(img, emotion_with_percent, (x, y-5),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 2)
            else:
                cv2.putText(img, emotion_with_percent, (x, y-5),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 6)
                cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 2)
       
    return img,count # Возвращаем изображение с нарисованными рамками вокруг распознанных лиц и эмоциями на лицах

def recognize_video(file_name,base_name):    
        cap = cv2.VideoCapture(file_name)
        pTime=0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            cTime=time.time()
            fps=1/(cTime-pTime)
            pTime=cTime
            img = copy.deepcopy(frame)
            img,count=MakePred(img)
            cv2.imshow(base_name, img)
            if cv2.waitKey(1) & 0xff== ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
    def initUI(self):

        app_icon = QIcon("C:/Users/lo297rd/Desktop/Face_Exp/CNN/icon.png")
        self.setWindowIcon(app_icon)

        self.setWindowTitle("FaceEmotionDetect")
        self.setGeometry(100, 100, 800, 600)

        self.setMinimumSize(QSize(800, 600))
        self.setMaximumSize(QSize(800, 600))
        self.setFixedSize(800, 600)

        self.button1 = QPushButton("Загрузить фотографию", self) 
        self.button1.setGeometry(QRect(QPoint(100, 100), QSize(250, 50)))        
        
        self.button2 = QPushButton("Распознать через веб-камеру", self)
        self.button2.setGeometry(QRect(QPoint(100, 200), QSize(250, 50)))
        
        self.button3 = QPushButton("Загрузить видео", self)
        self.button3.setGeometry(QRect(QPoint(100, 300), QSize(250, 50)))

        self.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 16px;
                border-radius: 5px;
                border: none;
            }

            QPushButton:hover {
                background-color: #3e8e41;
            }
        """)

        gradient = QLinearGradient(0, 0, 0, self.height())
        gradient.setColorAt(0.0, QColor(98, 218, 177))
        gradient.setColorAt(1.0, QColor(46, 196, 182))
        brush = QBrush(gradient)
        self.setAutoFillBackground(True)
        p = self.palette()
        p.setBrush(QPalette.ColorRole.Window, brush)
        self.setPalette(p)
             
        self.button1.clicked.connect(self.load_image)
        self.button2.clicked.connect(self.start_web)
        self.button3.clicked.connect(self.load_video)
    
    def start_web(self):
        recognize_video(0,"Web-camera")

    def load_video(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите файл",
            "",
            "Файлы видео (*.avi *.mp4 *.mov);;Все файлы (*)"
        )# Открываем диалоговое окно для выбора файла изображения
        if file_name:
            base_name=os.path.basename(file_name)
            recognize_video(file_name,base_name)
        
    def load_image(self):# Обработка события нажатия на кнопку "Загрузить фотографию"
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите файл",
            "",
            "Файлы изображений (*.png *.xpm *.jpg *.bmp);;Все файлы (*)"
        )# Открываем диалоговое окно для выбора файла изображения
        if file_name:# Если был выбран файл, загружаем его и обрабатываем
            image = cv2.imread(file_name)# Загружаем изображение и обрабатываем его с помощью модуля opencv и функции Predict_Photo()
            
            if image.shape[0]>900:
                scale_percent = 60 
                width = int(image.shape[1] * scale_percent / 100) 
                height = int(image.shape[0] * scale_percent / 100)
            else:
                width=int(image.shape[1])
                height=int(image.shape[0])


            image,count=MakePred(image)# Обработка изображения        
            if count==0:
                QMessageBox.critical(self, "Ошибка", "Лицо не найдено на изображении")
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)# Преобразуем цветовую схему изображения
                print("original:",image.shape)

                image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA) # Преобразуем изображение в нужный формат и размер, чтобы отобразить его в окне
                height, width, channel = image.shape
                print("resized:",image.shape)
                img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
               
                cv2.imshow('img',img_rgb)
                cv2.waitKey(0)
   
app = QApplication([])
window = MainWindow()
window.show()
app.exec()

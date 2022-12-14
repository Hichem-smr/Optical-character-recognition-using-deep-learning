import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QDialog, QApplication, QFileDialog
from PyQt5.uic import loadUi
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import arabic_reshaper
from bidi.algorithm import get_display


model=load_model('Reworked_modelV3.h5')

Arabic_words = []
Arabic_words.append("الرضّاع")
Arabic_words.append("الخليج")
Arabic_words.append("نقة")
Arabic_words.append("شعّال")
Arabic_words.append("مارث")
Arabic_words.append("الشمّاخ")
Arabic_words.append("زنّوش")
Arabic_words.append("الدخّانية")
Arabic_words.append("الفايض")
Arabic_words.append("أكّودة")
Arabic_words.append("سبعة آبار")
Arabic_words.append("سيدي ابراهيم الزهّار")
Arabic_words.append("المرناقية 20 مارس")
Arabic_words.append("شتاوة صحراوي")
Arabic_words.append("الفكّة")
Arabic_words.append("أوتيك")
Arabic_words.append("الفحص")
Arabic_words.append("الشرايع")
Arabic_words.append("حي الإنطلاقة")
Arabic_words.append("شواط")
Arabic_words.append("حي التضامن")



for word in range(len(Arabic_words)):
    Arabic_words[word] = arabic_reshaper.reshape(Arabic_words[word])
    Arabic_words[word] = Arabic_words[word][::-1]


class MainWindow(QDialog):
    def __init__(self):
        super(MainWindow,self).__init__()
        loadUi("gui.ui",self)
        self.browse.clicked.connect(self.browsefiles)
        self.predBtn.clicked.connect(self.encode)

    def browsefiles(self):
        self.fname = QFileDialog.getOpenFileName(self, 'Open file', '/home/schnoz/Documents/STOODIES/OCR/')
        self.filename.setText(self.fname[0])

    def encode(self):
        img_data_array = []

        image = cv2.imread(self.fname[0], cv2.IMREAD_UNCHANGED)
        image = cv2.resize(image, (40, 20), interpolation = cv2.INTER_AREA)
        image = image.astype('float')
        image /= 255
        img_data_array.append(image)
        pred = model.predict(np.array(img_data_array))
        print(pred[0])
        pred = np.argmax(pred[0])
        
        self.prediction.setText(get_display(Arabic_words[pred]) )
        print(get_display(Arabic_words[pred]))
        
        

app=QApplication(sys.argv)
mainwindow=MainWindow()
widget=QtWidgets.QStackedWidget()
widget.addWidget(mainwindow)
widget.setFixedWidth(400)
widget.setFixedHeight(300)
widget.show()
sys.exit(app.exec_())

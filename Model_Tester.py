from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QPushButton
from PyQt5.QtCore import Qt, QRect
from PyQt5.QtGui import QImage, QPainter, QPen, QPixmap, QFontDatabase
import sys, os, numpy as np, cv2 as cv


class app(QWidget):
    def __init__(self):
        # initialize lst_wb
        self.lst_wb = self.create_lst_wb()
        
        app = QApplication([])
        interface = self.main_gui()
        sys.exit(app.exec())
        
    def main_gui(self, parent=None):       
        super(app, self).__init__(parent)
        self.setWindowTitle('Detect Numeral!')
        self.setFixedSize(700, 500)

        self.left_pressed, self.last_point = False, None
        self.image = QImage(self.size(), QImage.Format_RGB32)
        self.image.fill(Qt.white)

        QFontDatabase.addApplicationFont('JosefinSans.ttf')
        QFontDatabase.addApplicationFont('RadioCanada.ttf')
        
        self.label1, self.label2, self.label3 = QLabel('Canvas Area!', self), QLabel('Prediction:', self), QLabel(self)
        
        # prediction string
        self.label3.setStyleSheet('font-family: Radio Canada; font-size: 40px')
        self.label3.move(380, 100)
        

        # canvas area
        self.label1.setStyleSheet('font-family: Josefin Sans; font-size:40px')
        self.label1.move(30, 0)
        
        # prediction
        self.label2.setStyleSheet('font-family: Josefin Sans; font-size:40px')
        self.label2.move(380, 55)
        
        # buttons
        self.predict_b = QPushButton('Predict!!', self)
        self.predict_b.move(30, 400)
        self.predict_b.clicked.connect(self.take_image_forP)
        
        self.clear_b = QPushButton('Clear Area', self)
        self.clear_b.move(110, 400)
        self.clear_b.clicked.connect(self.whiten)
        
        
        self.show()
        
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            if ((event.x() > 40) and (event.x() < 320)) and ((event.y() > 85) and (event.y() < 365)):
                self.left_pressed, self.last_point = True, event.pos()
            
    
    def mouseMoveEvent(self, event):
        if (event.buttons() == Qt.LeftButton) & self.left_pressed:
            if ((event.x() > 40) and (event.x() < 320)) and ((event.y() > 85) and (event.y() < 365)):

                painter = QPainter(self.image)
                painter.setRenderHint(QPainter.Antialiasing)
                
                painter.setPen(QPen(Qt.black, 20, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
                painter.drawLine(self.last_point, event.pos())
                
                self.last_point = event.pos()
                self.update()
    
    def whiten(self):
        self.image.fill(Qt.white)
        self.update()
    
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.left_pressed = False                   
    
    def paintEvent(self, event):
        canvas_paint = QPainter(self)
        canvas_paint.drawImage(self.rect(), self.image, self.image.rect())
        canvas_paint.setPen(QPen(Qt.black,  8, Qt.DashLine))
        canvas_paint.drawRect(40, 85, 300, 300)

        
    def take_image_forP(self):
        #screen grab
        ss = self.grab(QRect(45, 90, 280, 280))
        ss.save('shot.jpg')
        
        image = cv.imread('shot.jpg')
        image = cv.resize(image, (28, 28))
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        image = cv.bitwise_not(image)

        # forward prop 
        arr_fw = np.array(image).flatten() / 255
        output = self.forward_propagation(self.lst_wb, arr_fw)
        
        # updating the screen label
        self.prediction_str(output)
        self.update()
    

    def create_lst_wb(self):
        """ensure only weight bias files in folder and check for
        ipynb file at top (remove if present)"""
        files = os.listdir('learned_wb')
        files.sort()
        
        lst_wb, to_split = [], []
        for each in files:
            with open(f'learned_wb/{each}', 'r') as file:
                lines = file.readlines()
                for line in lines:
                    # doesn't matter for b as it has to have one column
                    to_split.append(line.split())
            
            arr_to_append = np.array(to_split).astype(float)
            lst_wb.append(arr_to_append)
            to_split = []

        lst_wb = [lst_wb[i:i+2] for i in range(0, len(lst_wb), 2)]
        return lst_wb
    
    def forward_propagation(self, lst_wb: list, X):
        def RelU(Z):
            return np.maximum(0, Z)
        
        def softmax(Z):
            return np.exp(Z)/sum(np.exp(Z))
            
        lst_za, prev_l = [], np.expand_dims(X, axis=1)
        for i in range(len(lst_wb) - 1):
            W, B = lst_wb[i][0], lst_wb[i][1]
            
            Z = W.dot(prev_l) + B
            A = RelU(Z)
            
            lst_za.append([Z, A])
            prev_l = A
            
        else:
            W_last, B_last = lst_wb[-1][0], lst_wb[-1][1]
            prev_a = lst_za[-1][1]
            
            Z = W_last.dot(prev_a) + B_last
            A = softmax(Z)
            
            lst_za.append([Z, A])
       
        out = lst_za[-1][1]
        return out
    
    def prediction_str(self, out):
        prediction = np.argmax(out, axis=0)[0]
        self.label3.clear()

        txt_output = f'{prediction}, {str(out[prediction][0] * 100)[:2]}%'
        self.label3.setText(txt_output)
        self.label3.adjustSize()



if __name__ == '__main__':
    app = app()


#%%

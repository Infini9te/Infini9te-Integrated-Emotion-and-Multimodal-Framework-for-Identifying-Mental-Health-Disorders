import sys
import cv2
import os
import api_head , api_gaze , api_mediapipe,json3_radar,api_emotion,radar,sequence_diagram
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QOpenGLWidget,QLabel, QPushButton
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPalette, QBrush, QMovie, QIcon
from PyQt5.QtCore import QTimer, QEventLoop, QThread, pyqtSignal, Qt, QSize
import time
import shutil
from ui2 import Ui_MainWindow  

os.chdir(r'C:\Users\NHRI\Desktop\情緒與專注力\UI')  

class MyForm(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setGeometry
        # # 設置 QMainWindow 的背景顏色
        # self.setStyleSheet("QMainWindow { background-color: #6a5acd; }")  # 這裡設置為淡灰色
        self.set_background_image('cutebg.jpg')

    def set_background_image(self, image_path):
        # 設置 QMainWindow 的背景圖片
        palette = QPalette()
        pixmap = QPixmap(image_path)  
        pixmap = pixmap.scaled(self.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
        brush = QBrush(pixmap)
        palette.setBrush(QPalette.Window, brush)
        self.setPalette(palette) 
        self.setAutoFillBackground(True)
        
        # 設置 QPushButton 的樣式
        button_style = """
        QPushButton {
            background: qradialgradient(cx: 0.3, cy: -0.4, fx: 0.3, fy: -0.4,
                    radius: 1.35, stop: 0.3 #2828FF, stop: 0.5 #4A4AFF, stop: 0.7 #6A6AFF, stop: 0.9 #7D7DFF);
            border: none;
            padding: 10px 20px;
            color: #FFFFFF;  /* 字體顏色 */
            font-family: Arial;  /* 字體 */
            font-size: 29px;  /* 字體大小 */
            text-align: center;
            text-decoration: none;
            font-weight: 900;  /* 字體粗細 */
            margin: 4px 2px;
            cursor: pointer;
            border-radius: 8px;
        }
        QPushButton:hover {
            background-color: qradialgradient(cx: 0.3, cy: -0.4, fx: 0.3, fy: -0.4,
                    radius: 1.35, stop: 0.3 #2828FF, stop: 0.5 #4A4AFF, stop: 0.7 #6A6AFF, stop: 0.9 #FFF7FB);  /* 當鼠標懸停時的顏色 */
        }
        """
        self.setStyleSheet(self.styleSheet() + button_style)

        # 將 centralwidget 設置為主窗口的中心部件
        self.setCentralWidget(self.ui.centralwidget)
        

        # 初始化定時器和 VideoCapture 對象
        self.timer1 = QTimer()
        self.timer1.timeout.connect(self.updateFrame1)
        self.timer2 = QTimer()
        self.timer2.timeout.connect(self.updateFrame2)
        self.cap1 = None
        self.cap2 = None

        # self.timer2 = QTimer()
        # self.timer2.timeout.connect(self.updateFrame2)
        # self.cap2 = None

        self.fileName1 = None  # 用於儲存 pushButton1 選擇的文件名
        self.fileName2 = None  # 用於儲存 openpic 選擇的文件名

        # 初始化 QGraphicsView 和 QGraphicsScene
        self.scene = QGraphicsScene()
        self.scene2 = QGraphicsScene()
        self.scene3 = QGraphicsScene()
        
        
        # 此處假設你的 UI 文件中有兩個 QGraphicsView widget
        self.graphicsView = self.ui.graphicsView
        self.graphicsView2 = self.ui.graphicsView2
        self.graphicsView3 = self.ui.graphicsView3
        
        self.graphicsView.setScene(self.scene)
        self.graphicsView2.setScene(self.scene2)
        self.graphicsView3.setScene(self.scene3)
        
        
        # 設置 QGraphicsView 的大小策略和固定大小
        # self.graphicsView.setFixedSize(550, 400)  # 設置具體的寬度和高度
        # self.graphicsView2.setFixedSize(800, 140)  # 設置具體的寬度和高度
        # self.graphicsView3.setFixedSize(550, 400)  # 設置具體的寬度和高度
        
        # 連接按鈕點擊事件與槽函數
        self.ui.pushButton3.clicked.connect(self.exportVideo)
        self.ui.pushButton4.clicked.connect(self.exportImage)  # 添加這行以連接匯出圖片按鈕的功能
        self.ui.pushButton5.clicked.connect(self.exportImage2)
        self.ui.pushButton6.clicked.connect(self.exportImage3)
        self.ui.pushButton_2.clicked.connect(self.back_to_video)
        self.ui.pushButton_3.clicked.connect(self.replay)
        self.ui.pushButton1_2.clicked.connect(self.openFile)
        self.ui.pushButton2_2.clicked.connect(self.start_a)
        self.ui.pushButton1_2.setIcon(QIcon('R.jpg'))
        self.ui.pushButton1_2.setIconSize(QSize(100,100))
        self.ui.pushButton2_2.setIcon(QIcon('2332.png'))
        self.ui.pushButton2_2.setIconSize(QSize(100,100))
        self.ui.pushButton_2.setIcon(QIcon('back.png'))
        self.ui.pushButton_2.setIconSize(QSize(100,100))
        self.ui.pushButton_3.setIcon(QIcon('replay.png'))
        self.ui.pushButton_3.setIconSize(QSize(75,75))
        
        new_button_style = """
        QPushButton {
            background-color: transparent;
            border: none;
            padding: 10px 20px;
            color: #000000;  /* 字體顏色 */
            font-family: Arial;  /* 字體 */
            font-size: 40px;  /* 字體大小 */
            text-align: center;
            text-decoration: none;
            font-weight: 900;  /* 字體粗細 */
            margin: 4px 2px;
            cursor: pointer;
            border-radius: 8px;
        }
        QPushButton:hover {
            background-color: qradialgradient(cx: 0.3, cy: -0.4, fx: 0.3, fy: -0.4,
                    radius: 1.35, stop: 0.3 #A6FFFF, stop: 0.5 #BBFFFF, stop: 0.7 #CAFFFF, stop: 0.9 #FFF7FB);  /* 當鼠標懸停時的顏色 */
        }
        """
        self.ui.pushButton1_2.setStyleSheet(new_button_style)
        self.ui.pushButton2_2.setStyleSheet(new_button_style)
        
        
        #將B3、4隱藏
        self.ui.pushButton5.setVisible(False)
        self.ui.pushButton6.setVisible(False)
        self.ui.pushButton3.setVisible(False)
        self.ui.pushButton4.setVisible(False)
        self.ui.pushButton_2.setVisible(False)
        self.ui.pushButton_3.setVisible(False)
        #隱藏介面
        self.graphicsView.setVisible(False)
        self.graphicsView2.setVisible(False)
        self.graphicsView3.setVisible(False)
        self.ui.widget.setVisible(False)
        self.ui.widget_2.setVisible(False)
        
        self.label_2 = self.ui.label_2  
        self.label_2 = QLabel(self.label_2)
        self.img = QImage('logo_NHRI_small.png')  
        pixmap = QPixmap.fromImage(self.img)  
        self.label_2.setPixmap(pixmap)
        self.label_2.setScaledContents(True)  
        self.label_2.setFixedSize(200, 200)  
        self.label_2.setVisible(True) 
        
        self.label_3 = self.ui.label_3  
        self.label_3 = QLabel(self.label_3)
        self.img = QImage('title.png')  
        pixmap = QPixmap.fromImage(self.img)  
        self.label_3.setPixmap(pixmap)
        self.label_3.setScaledContents(True)  
        self.label_3.setFixedSize(600, 200)  
        self.label_3.setVisible(True)
        
        
        self.label_4 = self.ui.label_4  
        self.label_4 = QLabel(self.label_4)
        self.img = QImage('sun.png')  
        pixmap = QPixmap.fromImage(self.img)  
        self.label_4.setPixmap(pixmap)
        self.label_4.setScaledContents(True)  
        self.label_4.setFixedSize(300, 300)  
        self.label_4.setVisible(True)
        
        
        
        # GIF 
        self.label = self.ui.label  
        self.label = QLabel(self.label)
        self.movie = QMovie('loading3.gif')
        self.label.setGeometry(0, 0, self.width(), self.height())
        self.movie.setScaledSize(self.size())
        self.label.setMovie(self.movie)
        self.movie.start()
        self.label.setVisible(False)  # 初始時隱藏

    def back_to_video(self):
        image_path2 = 'cutebg'
        palette = QPalette()
        pixmap = QPixmap(image_path2)  
        pixmap = pixmap.scaled(self.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
        brush = QBrush(pixmap)
        palette.setBrush(QPalette.Window, brush)
        self.setPalette(palette) 
        self.setAutoFillBackground(True)
        
        # 將 centralwidget 設置為主窗口的中心部件
        self.setCentralWidget(self.ui.centralwidget)
        
        # 初始化定時器和 VideoCapture 對象
        self.timer1 = QTimer()
        self.timer1.timeout.connect(self.updateFrame1)
        self.timer2 = QTimer()
        self.timer2.timeout.connect(self.updateFrame2)
        self.cap1 = None
        self.cap2 = None

        # self.timer2 = QTimer()
        # self.timer2.timeout.connect(self.updateFrame2)
        # self.cap2 = None

        self.fileName1 = None  # 用於儲存 pushButton1 選擇的文件名
        self.fileName2 = None  # 用於儲存 openpic 選擇的文件名

        # 初始化 QGraphicsView 和 QGraphicsScene
        self.scene = QGraphicsScene()
        self.scene2 = QGraphicsScene()
        self.scene3 = QGraphicsScene()
        
        
        # 此處假設你的 UI 文件中有兩個 QGraphicsView widget
        self.graphicsView = self.ui.graphicsView
        self.graphicsView2 = self.ui.graphicsView2
        self.graphicsView3 = self.ui.graphicsView3
        
        self.graphicsView.setScene(self.scene)
        self.graphicsView2.setScene(self.scene2)
        self.graphicsView3.setScene(self.scene3)
        
        #將B3、4隱藏
        self.ui.pushButton5.setVisible(False)
        self.ui.pushButton6.setVisible(False)
        self.ui.pushButton3.setVisible(False)
        self.ui.pushButton4.setVisible(False)
        
        self.ui.pushButton_2.setVisible(False)
        self.ui.pushButton_3.setVisible(False)
        self.ui.pushButton1_2.setVisible(True)
        self.ui.pushButton2_2.setVisible(True)
        
        #隱藏介面
        self.graphicsView.setVisible(False)
        self.graphicsView2.setVisible(False)
        self.graphicsView3.setVisible(False)
        self.ui.widget.setVisible(False)
        self.ui.widget_2.setVisible(False)
        new_img = QImage('sun.png')  
        new_pixmap = QPixmap.fromImage(new_img)
        self.label_4.setPixmap(new_pixmap)
        self.label_4.setVisible(True)
    
        
    def openFile(self):
        # 打開文件選擇對話框，選擇第一個影片文件
        options = QFileDialog.Options()
        self.fileName1, _ = QFileDialog.getOpenFileName(self, "選擇文件", "", "All Files (*);;Video Files (*.mp4 *.avi)", options=options)
        if self.fileName1:
            self.ui.widget_2.setVisible(True)
            self.startVideo2(self.fileName1)
            print("已選擇:", self.fileName1)
            
    class AnalyzeThread(QThread):
        finished = pyqtSignal()
        
        def __init__(self, parent):
            super().__init__()
            self.parent = parent
            
        def run(self):
            self.startanalyze()
            self.finished.emit()

        def startanalyze(self):
            api_head.processed_head(self.parent.fileName1)
            api_mediapipe.processed_mediapipe(self.parent.fileName1)
            api_gaze.processed_gaze(self.parent.fileName1)
            api_emotion.processed_emotion(self.parent.fileName1)
            sequence_diagram.main()
            self.parent.abc = "abc.jpg"
            json3_radar.main()
            self.parent.radar_chart = "radar_chart.jpg"
            radar.radar()
            self.parent.radar_chart2 = 'radar_chart2.jpg'
            self.parent.emotion_outout = "emotion_outout.mp4"

    def start_a(self):
        image_path3 = 'loading3.gif'
        palette = QPalette()
        pixmap = QPixmap(image_path3)  
        pixmap = pixmap.scaled(self.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
        brush = QBrush(pixmap)
        palette.setBrush(QPalette.Window, brush)
        self.setPalette(palette) 
        self.setAutoFillBackground(True)
        
        self.label_4.setVisible(False)
        self.ui.pushButton1_2.setVisible(False)
        self.ui.pushButton2_2.setVisible(False)
        self.label.setVisible(True)
        self.ui.widget_2.setVisible(False)
        # image_path = '20231004.jpg'
        # self.setStyleSheet(f"QMainWindow {{ background-image: url({image_path}); background-repeat: no-repeat; background-position: center; }}")
        loop = QEventLoop()
        QTimer.singleShot(0, loop.quit)
        loop.exec_()
        self.analyze_thread = self.AnalyzeThread(self)
        self.analyze_thread.finished.connect(self.on_analysis_finished)
        self.analyze_thread.start()
                    
    
    def replay(self):
        self.startVideo1(self.emotion_outout)
    
    def on_analysis_finished(self):
        image_path2 = 'moon1'
        palette = QPalette()
        pixmap = QPixmap(image_path2)  
        pixmap = pixmap.scaled(self.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
        brush = QBrush(pixmap)
        palette.setBrush(QPalette.Window, brush)
        self.setPalette(palette) 
        self.setAutoFillBackground(True)
        self.ui.widget.setVisible(True)
        
        if self.emotion_outout:
            print("第一個文件:", self.emotion_outout)
            self.startVideo1(self.emotion_outout)
        if self.radar_chart2:
            print("選擇的圖片文件:", self.radar_chart2)
            self.displayImage(self.radar_chart2)
        if self.radar_chart:
            print("選擇的圖片文件:", self.radar_chart)
            self.displayImage3(self.radar_chart)
        if self.abc:
            print("選擇的圖片文件:", self.abc)
            self.displayImage2(self.abc)
        else:
            print("尚未選擇任何文件")

        # 將B3、4顯示
        self.label.setVisible(False)
        self.graphicsView.setVisible(True)
        self.graphicsView2.setVisible(True)
        self.graphicsView3.setVisible(True)
        self.ui.pushButton3.setVisible(True)
        self.ui.pushButton4.setVisible(True)
        self.ui.pushButton5.setVisible(True)
        self.ui.pushButton6.setVisible(True)
        self.ui.pushButton_2.setVisible(True)
        self.ui.pushButton_3.setVisible(True)
        new_img = QImage('moon.png')  
        new_pixmap = QPixmap.fromImage(new_img)
        self.label_4.setPixmap(new_pixmap) 
        self.label_4.setVisible(True)
        
        
    def displayImage(self, fileName):
        # 將選擇的圖片加載為 QImage
        q_img = QImage(fileName)
        
        if q_img.isNull():
            print("Error: 無法加載圖片")
            return
        
        # 在 QGraphicsView 中顯示 QImage
        self.scene.clear()
        pixmap = QPixmap.fromImage(q_img)
        self.scene.addItem(QGraphicsPixmapItem(pixmap))

    def displayImage2(self, fileName):
        # 將選擇的圖片加載為 QImage
        q_img = QImage(fileName)
        
        if q_img.isNull():
            print("Error: 無法加載圖片")
            return
        
        # 在第二個 QGraphicsView 中顯示 QImage
        self.scene2.clear()
        pixmap = QPixmap.fromImage(q_img)
        self.scene2.addItem(QGraphicsPixmapItem(pixmap))
        
    def displayImage3(self, fileName):
        # 將選擇的圖片加載為 QImage
        q_img = QImage(fileName)
        
        if q_img.isNull():
            print("Error: 無法加載圖片")
            return
        
        # 在第二個 QGraphicsView 中顯示 QImage
        self.scene3.clear()
        pixmap = QPixmap.fromImage(q_img)
        self.scene3.addItem(QGraphicsPixmapItem(pixmap))

    def exportImage(self):
        # 打開文件保存對話框
        file_name, _ = QFileDialog.getSaveFileName(self, "保存圖片", "", "PNG Files (*.png);;JPEG Files (*.jpg)")
        if file_name:
            image_path = 'radar_chart2.jpg'
            shutil.copy(image_path, file_name)

    def exportImage2(self):
        # 打開文件保存對話框
        file_name, _ = QFileDialog.getSaveFileName(self, "保存圖片", "", "PNG Files (*.png);;JPEG Files (*.jpg)")
        if file_name:
            image_path = 'abc.jpg'
            shutil.copy(image_path, file_name)

    
    def exportImage3(self):
        # 打開文件保存對話框
        file_name, _ = QFileDialog.getSaveFileName(self, "保存圖片", "", "PNG Files (*.png);;JPEG Files (*.jpg)")
        if file_name:
            image_path = 'radar_chart.jpg'
            shutil.copy(image_path, file_name)
            
    def exportVideo(self):
        # 導出影片，打開文件保存對話框
        file_name, _ = QFileDialog.getSaveFileName(self, "保存影片", "", "MP4 Files (*.mp4);;AVI Files (*.avi)")
        if file_name:
            video_path = 'emotion_outout.mp4'
            shutil.copy(video_path, file_name)
    def startVideo1(self, A):
        self.cap1 = cv2.VideoCapture(A)
        if not self.cap1.isOpened():
            print("Error: 無法打開第一個影片文件")
            return
        self.timer1.start(30)  # 設置定時器以每30毫秒更新一次
        
    def startVideo2(self, B):
        self.cap2 = cv2.VideoCapture(B)
        if not self.cap2.isOpened():
            print("Error: 無法打開第二個影片文件")
            return
        self.timer2.start(30)  # 設置定時器以每30毫秒更新一次

    def updateFrame1(self):
        ret, frame = self.cap1.read()
        if not ret:
            self.timer1.stop()
            self.cap1.release()
            return

        # 將 frame 轉換為 QImage
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # 在第一個 openGLWidget 中顯示 QImage
        self.ui.openGLWidget1.updateImage(q_img)
        
    def updateFrame2(self):
        ret, frame = self.cap2.read()
        if not ret:
            self.timer2.stop()
            self.cap2.release()
            return

        # 將 frame 轉換為 QImage
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # 在第一個 openGLWidget 中顯示 QImage
        self.ui.openGLWidget2.updateImage(q_img)

class VideoWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image = None

    def updateImage(self, image):
        self.image = image
        self.update()

    def paintEvent(self, event):
        if self.image:
            painter = QPainter(self)
            rect = self.rect()

            # 計算長寬高
            widget_aspect_ratio = rect.width() / rect.height()
            image_aspect_ratio = self.image.width() / self.image.height()

            # 調整
            if widget_aspect_ratio > image_aspect_ratio:
                new_height = rect.height()
                new_width = int(new_height * image_aspect_ratio)
            else:
                new_width = rect.width()
                new_height = int(new_width / image_aspect_ratio)

            x_offset = (rect.width() - new_width) // 2
            y_offset = (rect.height() - new_height) // 2
            target_rect = rect.adjusted(x_offset, y_offset, -x_offset, -y_offset)
            painter.drawImage(target_rect, self.image)
            

class GIFLabel(QLabel):
    def __init__(self, gif_path, width, height, parent=None):
        super().__init__(parent)
        self.setFixedSize(width, height)
        self.movie = QMovie(gif_path)
        self.movie.setScaledSize(self.size())
        self.setMovie(self.movie)
        self.movie.start()

    def resizeEvent(self, event):
        # 在窗口大小改變時調整 GIF 大小
        self.movie.setScaledSize(self.size())
        super().resizeEvent(event)
        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    form = MyForm()
    # form = Form2()
    
    # 初始化兩個 VideoWidget
    form.ui.openGLWidget1 = VideoWidget(form.ui.widget)
    form.ui.openGLWidget2 = VideoWidget(form.ui.widget_2)
    # form.ui.openGLWidget1.setGeometry(50, 20, 800, 400)  
    form.ui.openGLWidget1.setFixedSize(1000, 600) # 設置第一個 VideoWidget 的位置和大小
    form.ui.openGLWidget2.setFixedSize(1000, 600)

    # form.ui.openGLWidget2 = VideoWidget(form.ui.widget)
    # form.ui.openGLWidget2.setGeometry(50, 430, 800, 400)  # 設置第二個 VideoWidget 的位置和大小

    form.show()
    sys.exit(app.exec_())

    

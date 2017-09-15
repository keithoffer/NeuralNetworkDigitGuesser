# PyQt is GPL v3 licenced
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget, QProgressBar
from PyQt5.QtCore import Qt, pyqtSignal, QSize, QPoint
from PyQt5.QtGui import QImage, qRgb, QPainter, QPen
# Python standard library is PSF licenced
import sys
# Numpy is under the modified-BSD licence
import numpy as np
# Pillow is under the PIL Software License, which is similar to the MIT license
from PIL import Image
# SciPy is under the modified-BSD licence
from scipy.ndimage.measurements import center_of_mass
# Files from this project
from neural_net.own.neural_network import NeuralNetwork

version = '1.0.0'

# This class is based in part on the scribble example that came with PyQt4
class DrawArea(QWidget):
    def __init__(self, parent=None):
        super(DrawArea, self).__init__(parent)

        self.neural_network = NeuralNetwork(load_from_file='neural_net/own/neural_network.npz')

        imageSize = QSize(560, 560)
        self.image = QImage(imageSize, QImage.Format_Grayscale8)
        self.lastPoint = QPoint()

    def clear_image(self):
        self.image.fill(qRgb(255, 255, 255))
        self.clear_estimate.emit()
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.lastPoint = event.pos()

    def mouseMoveEvent(self, event):
        if (event.buttons() & Qt.LeftButton):
            self.draw_line_to(event.pos())
            ptr = self.image.bits()
            ptr.setsize(self.image.byteCount())
            array = 255 - np.array(ptr).reshape(560, 560)
            cropped_arr = self.crop_image(array,0)
            # The following is the pre-processing done to numerals in the MNIST dataset, so we do it here to match
            # our training data
            i = Image.fromarray(cropped_arr,'L')
            i = i.resize((20, 20), Image.ANTIALIAS)
            arr = np.asarray(i)
            new_arr = np.zeros((28,28))
            new_arr[4:24,4:24] = arr
            (cx, cy) = center_of_mass(new_arr)
            dx = int(14-cx)
            dy = int(14-cy)
            new_arr = np.roll(new_arr, dx,axis=0)
            new_arr = np.roll(new_arr, dy, axis=1)
            # Now we process our image for input into the neural network
            new_arr_flattened = np.array(new_arr.reshape(784),dtype=np.uint8)
            input = np.abs(new_arr_flattened) / 255.0 * 0.99 + 0.01
            result = self.neural_network.query(input)
            self.new_estimate.emit(result)

    def crop_image(self, img, threshold=0):
        """
        This function crops the image to a bounding box that has a fixed aspect ratio of 1:1
        :param img: Array to crop
        :param threshold: Lower threshold to consider the pixel 'drawn in'
        :return: cropped image
        """
        mask = img > threshold
        cols = mask.any(0)
        rows = mask.any(1)
        # Find the first row and column where part of the number was drawn
        for col_index, col_value in enumerate(cols):
            if col_value:
                break
        for row_index, row_value in enumerate(rows):
            if row_value:
                break
        width = np.sum(mask.any(0))
        height = np.sum(mask.any(1))
        box_width = max((width,height))
        return img[row_index:row_index+box_width,col_index:col_index+box_width]

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.draw_line_to(event.pos())
        if event.button() == Qt.RightButton:
            self.clear_image()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawImage(event.rect(), self.image)

    def draw_line_to(self, endPoint):
        painter = QPainter(self.image)
        painter.setPen(QPen(Qt.black, 8,Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        painter.drawLine(self.lastPoint, endPoint)

        self.update()
        self.lastPoint = QPoint(endPoint)

    new_estimate = pyqtSignal(object)
    clear_estimate = pyqtSignal()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle('Number Guesser ' + version)
        # This layout is about the most complicated I'd want to have without using a QDesigner generated file
        # Perhaps it would be better to use QDesiger anyway
        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)
        self.main_ver_layout = QVBoxLayout(self)
        self.centralWidget().setLayout(self.main_ver_layout)

        self.hor_layout = QHBoxLayout()
        self.ver_layout_progressbars = QVBoxLayout(self)

        self.draw_area = DrawArea(self)
        # Force the area to be an unchanging 560 x 560 canvas
        self.draw_area.setMaximumWidth(560)
        self.draw_area.setMaximumHeight(560)
        self.draw_area.setMinimumWidth(560)
        self.draw_area.setMinimumHeight(560)
        self.draw_area.clear_image()

        self.digit_label = QLabel('Current estimate = <span style="color:blue">N/A</span>')
        self.digit_label.setAlignment(Qt.AlignHCenter | Qt.AlignCenter)
        font = self.digit_label.font()
        font.setPointSize(24)
        self.digit_label.setFont(font)

        self.progress_bar_list = []
        progress_bar_label = QLabel('Relative digit confidence')
        progress_bar_label.setAlignment(Qt.AlignHCenter)
        progress_bar_label.setMaximumHeight(20)
        self.ver_layout_progressbars.addWidget(progress_bar_label)
        for n in range(10):
            new_progress_bar = QProgressBar(self)
            label = QLabel(f'{str(n)} : ')
            new_progress_bar.setValue(0)
            self.progress_bar_list.append(new_progress_bar)
            bar_hor_layout = QHBoxLayout()
            bar_hor_layout.addWidget(label)
            bar_hor_layout.addWidget(new_progress_bar)
            self.ver_layout_progressbars.addLayout(bar_hor_layout)

        self.hor_layout.addWidget(self.draw_area)
        self.hor_layout.addLayout(self.ver_layout_progressbars)
        self.main_ver_layout.addWidget(self.digit_label)
        self.main_ver_layout.addLayout(self.hor_layout)

        self.draw_area.new_estimate.connect(self.update_estimate)
        self.draw_area.clear_estimate.connect(self.clear_estimate)

        self.show()

    def clear_estimate(self):
        self.digit_label.setText('Current estimate = <span style="color:blue">N/A</span>')
        for bar in self.progress_bar_list:
            bar.setValue(0)

    def update_estimate(self,estimate):
        total = np.sum(estimate)
        self.digit_label.setText(f'Current estimate = <span style="color:blue">{np.argmax(estimate)}</span>')
        for index,bar in enumerate(self.progress_bar_list):
            bar.setValue(estimate[index]/total*100)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    GUI = MainWindow()
    sys.exit(app.exec())
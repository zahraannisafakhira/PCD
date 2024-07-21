import sys
import cv2
import numpy as np
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUi
from matplotlib import pyplot as plt

class ShowImage(QMainWindow):
    def __init__(self):
        super(ShowImage, self).__init__()
        loadUi('showgui.ui', self)
        self.Image = None
        self.button_loadCitra.clicked.connect(self.fungsi)
        self.button_prosesCitra.clicked.connect(self.grayscale)
        self.actionOperasi_Pencerahan.triggered.connect(self.brightness)
        self.actionSimple_Contrast.triggered.connect(self.contrast)
        self.actionContrast_Stretching.triggered.connect(self.contrastStretching)
        self.actionNegative_Image.triggered.connect(self.negativeImage)
        self.actionBiner_Image.triggered.connect(self.binaryImage)
        self.actionHistogram_Gray.triggered.connect(self.grayHistogram)
        self.actionHistogram_RGB.triggered.connect(self.rgbHistogram)
        self.actionHistogram_Equal.triggered.connect(self.equalHistogram)

        # Operasi Geomteri
        self.actionTranslasi.triggered.connect(self.translasi)
        self.action90_Derajat.triggered.connect(self.rotasi90derajat)
        self.actionZoom_In.triggered.connect(self.zoomIn)
        self.actionDimensi.triggered.connect(self.dimensi)
        self.actionCrop.triggered.connect(self.cropImage)

        # Operasi Atirmatika
        self.actionTambah_dan_Kurang.triggered.connect(self.aritmatika)

        # Operasi Boolean
        self.actionOperasi_AND.triggered.connect(self.operasiAND)

        self.actionKonvolusi_2D.triggered.connect(self.konvolusi2D)
        self.actionMean_Filter.triggered.connect(self.meanFilter)
        self.actionGauss_Filter.triggered.connect(self.gaussFilter)


    def fungsi(self):
        self.Image = cv2.imread('img.jpg')
        self.displayImage(1)

    def grayscale(self):
        if self.Image is not None:
            gray = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
            self.Image = gray
            self.displayImage(2)

    def brightness(self):
        try:
            self.Image = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        except:
            pass
        brightness = 80
        bright_img = cv2.convertScaleAbs(self.Image, alpha=1, beta=brightness)
        self.Image = bright_img
        self.displayImage(1)

    def contrast(self):
        try:
            self.Image = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        except:
            pass
        contrast = 1.7
        contrast_img = cv2.convertScaleAbs(self.Image, alpha=contrast, beta=0)
        self.Image = contrast_img
        self.displayImage(1)

    def contrastStretching(self):
        if self.Image is not None:
            min_val = np.min(self.Image)
            max_val = np.max(self.Image)
            stretched_img = cv2.normalize(self.Image, None, 0, 255, cv2.NORM_MINMAX)
            self.Image = stretched_img
            self.displayImage(1)

    def negativeImage(self):
        if self.Image is not None:
            negative_img = 255 - self.Image
            self.Image = negative_img
            self.displayImage(1)

    def binaryImage(self):
        if self.Image is not None:
            if len(self.Image.shape) == 3:  # Check if the image is colored
                gray = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
            else:
                gray = self.Image
            # Apply binary threshold
            threshold_value, binary_img = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            # Compare pixel values before and after thresholding
            H, W = gray.shape[:2]
            comparison_img = np.zeros((H, W, 3), np.uint8)
            for i in range(H):
                for j in range(W):
                    original_value = gray[i, j]
                    binary_value = binary_img[i, j]
                    comparison_img[i, j] = [original_value, binary_value, 0]
            self.Image = binary_img
            self.displayImage(1)

    def grayHistogram(self):
        if self.Image is not None:
            plt.hist(self.Image.ravel(), 255, [0, 255])
            plt.show()

    def rgbHistogram(self):
        if self.Image is not None:
            color = ('b', 'g', 'r')
            for i, col in enumerate(color):
                histo = cv2.calcHist([self.Image], [i], None, [256], [0, 256])
                plt.plot(histo, color=col)
            plt.xlim([0, 256])
            plt.show()

    def equalHistogram(self):
        if self.Image is not None:
            hist, bins = np.histogram(self.Image.flatten(), 256, [0, 256])
            cdf = hist.cumsum()
            cdf_normalized = cdf * hist.max() / cdf.max()
            cdf_m = np.ma.masked_equal(cdf, 0)
            cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
            cdf = np.ma.filled(cdf_m, 0).astype("uint8")
            self.displayImage(2)
            plt.plot(cdf_normalized, color="b")
            plt.hist(self.Image.flatten(), 256, [0, 256], color="r")
            plt.xlim([0, 256])
            plt.legend(("cdf", "histogram"), loc="upper left")
            plt.show()

    def translasi(self):
        h, w = self.Image.shape[:2]
        quarter_h,quarter_w = h/4,w/4
        T = np.float32([[1,0,quarter_w], [0,1,quarter_h]])
        img = cv2.warpAffine(self.Image,T,(w, h))
        self.Image = img
        self.displayImage(2)

    def rotasi(self, degree):
        h,w = self.Image.shape[:2]
        rotationMatrix = cv2.getRotationMatrix2D((w/2, h/2), degree, 0.7)
        cos = np.abs(rotationMatrix[0, 0])
        sin = np.abs(rotationMatrix[0, 1])
        nW = int((h*sin) + (w*cos))
        nH = int((h*cos) + (h*sin))

        rotationMatrix[0, 2] += (nW / 2) - w/2
        rotationMatrix[1, 2] += (nH / 2) - h/2

        rot_image = cv2.warpAffine(self.Image, rotationMatrix, (h, w))
        self.Image = rot_image
        self.displayImage(2)

    def rotasi90derajat(self):
        self.rotasi(90)

    def zoomIn(self):
        skala = 1
        resize_image = cv2.resize(self.Image, None, fx=skala, fy=skala, interpolation=cv2.INTER_CUBIC)
        cv2.imshow('Original', resize_image)
        cv2.imshow('Zoom In', resize_image)
        cv2.waitKey()

    def dimensi(self):
        resize_image = cv2.resize(self.Image, (900, 400), interpolation=cv2.INTER_AREA)
        cv2.imshow('original', self.Image)
        cv2.imshow('dimensi', resize_image)
        cv2.waitKey()

    def cropImage(self):
        start_row = 50
        end_row = 200
        start_col = 100
        end_col = 300
        crop_image = self.Image[start_row:end_row, start_col:end_col]
        cv2.imshow('original', self.Image)
        cv2.imshow('Crop Image', crop_image)
        cv2.waitKey()

    def aritmatika(self):
        image1 = cv2.imread('img.jpg', 0)
        image2 = cv2.imread('img.jpg', 0)
        image_tambah = image1 + image2
        image_kurang = image1 - image2
        cv2.imshow('Image 1 Original', image1)
        cv2.imshow('Image 2 Original', image2)
        cv2.imshow('Image Tambah', image_tambah)
        cv2.imshow('Image Kurang', image_kurang)
        cv2.waitKey()

    def operasiAND(self):
        image1 = cv2.imread('img.jpg', 1)
        image2 = cv2.imread('img.jpg', 1)
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
        operasi = cv2.bitwise_and(image1, image2)
        cv2.imshow('Image 1 Original', image1)
        cv2.imshow('Image 2 Original', image2)
        cv2.imshow('Image Operasi AND', operasi)
        cv2.waitKey()

    def konvolusi2D(self):
        kernel = np.array([
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ])
        img_input = cv2.imread('img.jpg')
        gray_image = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)
        img_out = cv2.filter2D(gray_image, -1, kernel)
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])
        cv2.imshow('Image belum konvolusi', img_input)
        plt.show()

    def meanFilter(self):
        mean = (1 / 9) * np.array([
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ])
        img_input = cv2.imread('img.jpg')
        gray_image = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)
        img_out = cv2.filter2D(gray_image, -1, mean)
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])
        cv2.imshow('Image belum konvolusi', img_input)
        plt.show()

    def gaussFilter(self):
        gauss = (1.0 / 345) * np.array([
            [1, 5, 7, 5, 1],
            [5, 20, 33, 20, 5],
            [7, 33, 55, 33, 7],
            [5, 20, 33, 20, 5],
            [1, 5, 7, 5, 1]
        ])
        img_input = cv2.imread('img.jpg')
        gray_image = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)
        img_out = cv2.filter2D(gray_image, -1, gauss)
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])
        cv2.imshow('Image belum konvolusi', img_input)
        plt.show()

    def displayImage(self, window=1):
        if self.Image is not None:
            qformat = QImage.Format_Indexed8
            if len(self.Image.shape) == 3:
                if self.Image.shape[2] == 4:
                    qformat = QImage.Format_RGBA8888
                else:
                    qformat = QImage.Format_RGB888
            img = QImage(self.Image, self.Image.shape[1], self.Image.shape[0], self.Image.strides[0], qformat)
            img = img.rgbSwapped()
            if window == 1:
                self.label.setPixmap(QPixmap.fromImage(img))
            elif window == 2:
                self.label_2.setPixmap(QPixmap.fromImage(img))

app = QtWidgets.QApplication(sys.argv)
window = ShowImage()
window.setWindowTitle('Show Image GUI')
window.show()
sys.exit(app.exec_())

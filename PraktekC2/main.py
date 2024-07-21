import sys

import cv2
import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUi
from matplotlib import pyplot as plt


class ShowImage(QMainWindow):
    def __init__(self):
        super(ShowImage, self).__init__()
        loadUi('showgui.ui', self)
        self.Image = None
        self.loadButton.clicked.connect(self.fungsi)
        self.grayButton.clicked.connect(self.grayscale)
        self.actionOperasi_Pencerahan.triggered.connect(self.brightness)
        self.actionSimple_Contrast.triggered.connect(self.contrast)
        self.actionContrast_Streching.triggered.connect(self.contrastStretching)
        self.actionNegative_Image.triggered.connect(self.negativeImage)
        self.actionBiner.triggered.connect(self.binaryImage)
        self.actionHistogram_Gray.triggered.connect(self.grayHistogram)
        self.actionHistogram_RGB.triggered.connect(self.rgbHistogram)
        self.actionHistogram_Equal.triggered.connect(self.equalHistogram)
        self.actionTranslasi.triggered.connect(self.translasi)
        self.actionRotate_90.triggered.connect(self.rotasi90derajat)
        self.actionRotate_45.triggered.connect(self.rotasi45derajat)
        self.actionRotate_180.triggered.connect(self.rotasi180derajat)
        self.actionRotate_min_45.triggered.connect(self.rotasimin45derajat)
        self.actionRotate_min_90.triggered.connect(self.rotasimin90derajat)
        self.actionZoom_in.triggered.connect(self.zoomIn)
        self.actionZoom_Out.triggered.connect(self.zoomOut)
        self.actionDimension.triggered.connect(self.dimensi)
        self.actionCrop.triggered.connect(self.cropImage)
        self.actionAdd.triggered.connect(self.arithmathicAdd)
        self.actionSubstract.triggered.connect(self.arithmathicSubstract)
        self.actionAnd.triggered.connect(self.booleanAnd)
        self.actionOr.triggered.connect(self.booleanOr)
        self.actionNot.triggered.connect(self.booleanNot)
        self.actionXor.triggered.connect(self.booleanXor)

    def fungsi(self):
        self.Image = cv2.imread('smmhealth.jpeg')
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
        quarter_h, quarter_w = h / 4, w / 4
        T = np.float32([[1, 0, quarter_w], [0, 1, quarter_h]])
        img = cv2.warpAffine(self.Image, T, (w, h))
        self.Image = img
        self.displayImage(2)

    def rotasi(self, degree):
        h, w = self.Image.shape[:2]
        rotationMatrix = cv2.getRotationMatrix2D((w / 2, h / 2), degree, 0.7)
        cos = np.abs(rotationMatrix[0, 0])
        sin = np.abs(rotationMatrix[0, 1])
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (h * sin))
        rotationMatrix[0, 2] += (nW / 2) - w / 2
        rotationMatrix[1, 2] += (nH / 2) - h / 2
        rot_image = cv2.warpAffine(self.Image, rotationMatrix, (h, w))
        self.Image = rot_image
        self.displayImage(2)

    def rotasi45derajat(self):
        self.rotasi(45)

    def rotasimin45derajat(self):
        self.rotasi(-45)

    def rotasi90derajat(self):
        self.rotasi(90)

    def rotasimin90derajat(self):
        self.rotasi(-90)

    def rotasi180derajat(self):
        self.rotasi(180)

    def zoomIn(self):
        skala = 0.5
        resize_Image = cv2.resize(self.Image, None, fx=skala, fy=skala, interpolation=cv2.INTER_CUBIC)
        cv2.imshow('original', self.Image)
        cv2.imshow('zoom In', resize_Image)
        cv2.waitKey()

    def zoomOut(self):
        skala = 0.50
        resize_image = cv2.resize(self.Image, None, fx=skala, fy=skala)
        cv2.imshow('original', self.Image)
        cv2.imshow('Zoom Out', resize_image)
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

    def arithmathicAdd(self):
        img1 = cv2.imread('smmhealth.jpeg', 0)
        img2 = cv2.imread('smmhealth.jpeg', 0)
        add_img = img1 + img2
        cv2.imshow('Image 1 Original', img1)
        cv2.imshow('Image 2 Original', img2)
        cv2.imshow('Image add', add_img)
        cv2.waitKey()

    def arithmathicSubstract(self):
        img1 = cv2.imread('smmhealth.jpeg', 0)
        img2 = cv2.imread('smmhealth.jpeg', 0)
        subtract = img1 - img2
        cv2.imshow('Image 1 Original', img1)
        cv2.imshow('Image 2 Original', img2)
        cv2.imshow('Image Subtract', subtract)
        cv2.waitKey()

    def booleanAnd(self):
        img1 = cv2.imread('smoll.jpg', 1)
        img2 = cv2.imread('smoll.jpg', 1)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        boolean_and = cv2.bitwise_and(img1, img2)
        cv2.imshow('Image 1 Original', img1)
        cv2.imshow('Image 2 Original', img2)
        cv2.imshow('Image Boolean Add', boolean_and)
        cv2.waitKey()

    def booleanOr(self):
        img1 = cv2.imread('smoll.jpg', 1)
        img2 = cv2.imread('smoll.jpg', 1)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        boolean_or = cv2.bitwise_or(img1, img2)
        cv2.imshow('Image 1 Original', img1)
        cv2.imshow('Image 2 Original', img2)
        cv2.imshow('Image Boolean Or', boolean_or)
        cv2.waitKey()

    def booleanNot(self):
        img1 = cv2.imread('smoll.jpg', 1)
        img2 = cv2.imread('smoll.jpg', 1)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        boolean_not = cv2.bitwise_not(img1, img2)
        cv2.imshow('Image 1 Original', img1)
        cv2.imshow('Image 2 Original', img2)
        cv2.imshow('Image Boolean Not', boolean_not)
        cv2.waitKey()

    def booleanXor(self):
        img1 = cv2.imread('smoll.jpg', 1)
        img2 = cv2.imread('smoll.jpg', 1)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        boolean_xor = cv2.bitwise_xor(img1, img2)
        cv2.imshow('Image 1 Original', img1)
        cv2.imshow('Image 2 Original', img2)
        cv2.imshow('Image Boolean Xor', boolean_xor)

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
window.setWindowTitle('ShowImageGUI')
window.show()
sys.exit(app.exec_())

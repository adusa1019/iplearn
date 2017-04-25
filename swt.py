# coding=utf-8

import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time


class stroke_width_transform():
    def __init__(self):
        pass

    def run(self, image):
        # 1. The Stroke Width Transform
        swt = self.calculate_stroke_width_transform(image)
        # 2. Finding Letter Candidates
        # 3. Grouping Letters into Text Lines
        pass

    def calculate_stroke_width_transform(self, image):
        # assumption: image given is BGR
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_shape = np.shape(gray_image)

        # get image edge(threshold values are temporary)
        canny_threshold_low = 32
        canny_threshold_high = 256
        edges = cv2.Canny(gray_image, canny_threshold_low, canny_threshold_high)

        # get gradients[rad](sobel_filter_size is temporary)
        sobel_filter_size = -1
        dx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, sobel_filter_size)
        dy = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, sobel_filter_size)
        angles = np.arctan2(dy, dx)

        # initialize SWT
        swt = np.zeros(image_shape)
        swt.fill(np.infty)

        return swt


if __name__ == '__main__':
    path = glob.glob('C:/Users/mueda/PycharmProjects/StrokeWidthTransform/test/images/*.jpg')
    img = cv2.imread(path[0], 1)
    tmp = stroke_width_transform()
    tmp.run(img)

# coding=utf-8

import cv2
import glob
import math
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
        print(np.shape(image))
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_shape = np.shape(gray_image)

        # get image edge(threshold values are temporary)
        canny_threshold_low = 32
        canny_threshold_high = 256
        edges = cv2.Canny(gray_image, canny_threshold_low, canny_threshold_high)
        edge_coordinates = np.transpose(np.nonzero(edges))

        # get gradients[rad](sobel_filter_size is temporary)
        sobel_filter_size = -1
        dx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, sobel_filter_size)
        dy = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, sobel_filter_size)
        ex = dx / np.sqrt(dx ** 2 + dy ** 2)
        ey = dy / np.sqrt(dx ** 2 + dy ** 2)
        directions = np.arctan2(dy, dx)

        # initialize SWT
        swt = np.zeros(image_shape)
        swt.fill(np.infty)

        # print([639, 461] in edge_coordinates)
        # """
        # x:horizontal, y: vertical
        rays = []
        for y, x in edge_coordinates:
            ray = []
            ray.append([y, x])
            curr_x, curr_y, i = x, y, 0
            curr_ex, curr_ey = ex[y][x], ey[y][x]
            while True:
                i += 1
                next_x = math.floor(x + curr_ex * i)
                next_y = math.floor(y + curr_ey * i)

                if next_x != curr_x or next_y != curr_y:
                    # we have moved to the next pixel!
                    try:
                        ray.append((next_x, next_y))
                        # edgeかつ法線ベクトルのなす角が  between frac{5}{6}\pi and frac{7}{6}\pi
                        if [next_y, next_x] in edge_coordinates and np.dot(np.array([curr_ex, curr_ey]),
                                np.array([ex[next_y][next_x], ey[next_y][next_x]])) < -math.sqrt(3) / 2.0:
                            thickness = math.sqrt((next_x - x) * (next_x - x) + (next_y - y) * (next_y - y))
                            print(ray)
                            for (rp_x, rp_y) in ray:
                               swt[rp_y, rp_x] = min(thickness, swt[rp_y, rp_x])
                            break
                    except IndexError:
                        # reached image boundary
                        break
                    curr_x = next_x
                    curr_y = next_y
            # """
        return swt


if __name__ == '__main__':
    img = cv2.imread("036.jpg", 1)
    tmp = stroke_width_transform()
    tmp.run(img)

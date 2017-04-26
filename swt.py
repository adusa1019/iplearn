# coding=utf-8

import cv2
import math
import numpy as np
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

    @staticmethod
    def calculate_stroke_width_transform(image):
        # assumption: given image has BGR expression
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_shape = np.shape(gray_image)

        # get image edge(threshold values are temporary)
        canny_threshold_low = 100
        canny_threshold_high = 300
        edges = cv2.Canny(gray_image, canny_threshold_low, canny_threshold_high)
        edge_coordinates = np.transpose(np.nonzero(edges))

        # get gradients(sobel_filter_size is temporary)
        sobel_filter_size = -1
        dx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, sobel_filter_size)
        dy = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, sobel_filter_size)
        ex = dx / np.sqrt(dx ** 2 + dy ** 2)
        ey = dy / np.sqrt(dx ** 2 + dy ** 2)

        # initialize SWT
        swt = np.empty(image_shape)
        swt.fill(np.infty)

        # x:horizontal, y: vertical
        start = time.time()
        rays = []
        for y, x in edge_coordinates:
            if dx[y][x] == dy[y][x] == 0:
                continue
            ray = []
            curr_x, curr_y, i = x, y, 0
            base_ex, base_ey = ex[y][x], ey[y][x]

            ray.append([y, x])

            # for i in range(max(image_shape) * 2):
            for i in range(10):
                next_x = math.floor(x - base_ex * i)
                next_y = math.floor(y - base_ey * i)

                # current pixel == the next pixel
                if next_x == curr_x and next_y == curr_y:
                    continue
                # the next pixel is image boundary
                if next_y < 0 or image_shape[0] <= next_y or next_x < 0 or image_shape[1] <= next_x:
                    break

                ray.append([next_y, next_x])
                curr_x, curr_y = next_x, next_y

                if edges[next_y][next_x] == 0:
                    continue
                if dx[next_y][next_x] == dy[next_y][next_x] == 0:
                    continue
                epsilon = 0.1
                if np.dot([base_ey, base_ex], [ey[next_y][next_x], ex[next_y][next_x]]) > -1 + epsilon:
                    continue

                rays.append(ray)
                break
        # first time
        for ray in rays:
            stroke_width = np.linalg.norm(np.array([ray[0][0], ray[0][1]]) - np.array([ray[-1][0], ray[-1][1]]))
            for r_y, r_x in ray:
                swt[r_y, r_x] = min(stroke_width, swt[r_y, r_x])
        # second time
        for ray in rays:
            median = np.median([swt[r_y][r_x] for r_y, r_x in ray])
            for r_y, r_x in ray:
                swt[r_y, r_x] = min(median, swt[r_y, r_x])

        print(time.time() - start)
        # cv2.imshow("edge", edges)
        # cv2.imshow("swt", swt)
        # cv2.waitKey(0)
        return swt


if __name__ == '__main__':
    img = cv2.imread("036.jpg", 1)
    tmp = stroke_width_transform()
    tmp.run(img)

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
        swt, edge = self.calculate_stroke_width_transform(image)
        # 2. Finding Letter Candidates
        connect_components = self.connect_components(swt)
        swts, heights, widths, topleft_pts, images = self.find_letters(swt, connect_components)
        # 3. Grouping Letters into Text Lines
        word_images = self.find_words(swts, heights, widths, topleft_pts, images)
        return edge, word_images, heights, widths, topleft_pts

    @staticmethod
    def get_derivative(image, low=100, high=300):
        sobel_filter_size = -1
        edge = cv2.Canny(image, low, high)
        dx = cv2.Sobel(image, cv2.CV_64F, 1, 0, sobel_filter_size)
        dy = cv2.Sobel(image, cv2.CV_64F, 0, 1, sobel_filter_size)
        return edge, np.arctan2(dy, dx) * edge / np.max(edge)

    @staticmethod
    def is_valid_index(shape, x, y):
        return 0 <= y < shape[0] and 0 <= x < shape[1]

    def calculate_stroke_width_transform(self, image, edge=None, theta=None, direction=1):
        if image.ndim == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image

        # get image edge(threshold values are temporary)
        if edge is None and theta is None:
            edge, theta = self.get_derivative(gray_image)
        edge_coordinates = np.transpose(edge.nonzero())

        # initialize SWT
        swt = np.empty(gray_image.shape)
        swt.fill(np.infty)
        # x:horizontal, y: vertical
        start = time.time()
        rays = []
        stroke_widths = []
        for y, x in edge_coordinates:
            ray, stroke_width = self.cast_ray(x, y, theta, edge)
            if ray:
                rays.append(ray)
                stroke_widths.append(stroke_width)
        # first time
        for ray, stroke_width in zip(rays, stroke_widths):
            for r_y, r_x in ray:
                swt[r_y, r_x] = min(stroke_width, swt[r_y, r_x])
        # second time
        for ray in rays:
            median = np.median([swt[r_y][r_x] for r_y, r_x in ray])
            for r_y, r_x in ray:
                swt[r_y, r_x] = min(median, swt[r_y, r_x])
                # swt[r_y, r_x] = median
        print(time.time() - start)
        swt[np.isinf(swt)] = 0
        return swt, edge

    def cast_ray(self, start_x, start_y, theta, edge, max_length=25, direction=1, epsilon=1):
        ray = [(start_y, start_x)]
        curr_x, curr_y = start_x, start_y
        ray_direction = theta[start_y][start_x]
        base_ex, base_ey = np.cos(ray_direction), np.sin(ray_direction)

        for ray_length in range(1, max_length):
            next_x = math.floor(start_x - base_ex * ray_length * direction)
            next_y = math.floor(start_y - base_ey * ray_length * direction)
            if next_x == curr_x and next_y == curr_y:
                continue
            if not self.is_valid_index(theta.shape, next_x, next_y):
                return None, None
            ray.append((next_y, next_x))
            curr_x, curr_y = next_x, next_y
            if edge[next_y][next_x] == 0:
                continue
            opposite_direction = theta[next_y][next_x]
            if math.pi - epsilon < abs(ray_direction - opposite_direction) < math.pi + epsilon:
                return ray, ray_length
        return None, None

    # ToDo:Tooooooooo late
    def connect_components(self, swt):
        start = time.time()
        swt_coordinates = np.transpose(swt.nonzero())
        label_map = np.zeros(shape=swt.shape)
        next_label = 0
        neighbors_coordinates = [
            [
                (y - 1, x - 1),  # northwest
                (y - 1, x),  # north
                (y - 1, x + 1),  # northeast
                (y, x - 1)  # west
            ] for y, x in swt_coordinates
        ]
        ratio_threshold = 3.0
        for current, neighbors in zip(swt_coordinates, neighbors_coordinates):
            next_label += 1
            label_map[current[0], current[1]] = next_label
            swt_current = swt[current[0], current[1]]
            label = next_label
            for neighbor in neighbors:
                if not self.is_valid_index(swt.shape, neighbor[1], neighbor[0]):
                    continue
                if swt[neighbor] == 0:
                    continue
                swt_neighbor = swt[neighbor]
                if 1.0 / ratio_threshold < 1.0 * swt_neighbor / swt_current < ratio_threshold:
                    label = min(label_map[neighbor], label)
                    label_map[current[0], current[1]] = label
                    label_map[neighbor] = label
        # labels = [(i, np.transpose(np.nonzero(label_map == i))) for i in range(1, next_label) if np.any(label_map == i)]
        # """
        labels = dict([])
        for i in range(1, next_label):
            if np.all(label_map != i):
                continue
            # labels[str(i)] = [(r, c) for r, c in np.transpose(np.nonzero(label_map == i))]
            labels[str(i)] = np.transpose(np.nonzero(label_map == i))
        # """
        print(time.time() - start)
        return labels

    @staticmethod
    def find_letters(swt, shapes):
        start = time.time()
        swts = []
        heights = []
        widths = []
        topleft_pts = []
        images = []
        for label, layer in shapes.items():
            # print(layer)
            ys, xs = np.transpose(layer)
            east, west, south, north = max(xs), min(xs), max(ys), min(ys)
            width, height = east - west, south - north

            if width < 8 or height < 8:
                continue
            if width / height > 10 or height / width > 10:
                continue
            diameter = math.sqrt(width**2 + height**2)
            median_swt = np.median([swt[r, c] for r, c in layer])
            # if diameter / median_swt > 10:
            #     continue
            if width / swt.shape[1] > 0.4 or height / swt.shape[0] > 0.4:
                continue

            # we use log_base_2 so we can do linear distance comparison later using k-d tree
            # ie, if log2(x) - log2(y) > 1, we know that x > 2*y
            # Assumption: we've eliminated anything with median_swt == 1
            swts.append([math.log(median_swt, 2)])
            heights.append([math.log(height, 2)])
            topleft_pts.append(np.asarray([north, west]))
            widths.append(width)
            images.append(layer)
        print(time.time() - start)
        return swts, heights, widths, topleft_pts, images

    @staticmethod
    def find_words(swts, heights, widths, topleft_pts, images):
        word_images = None
        return word_images or images


if __name__ == '__main__':
    img = cv2.imread("036.jpg", 1)
    tmp = stroke_width_transform()
    edge, mask = tmp.run(img)
    # for component in mask_components:
    #     mask = np.zeros(shape=swt.shape)
    #     for r, c in component:
    #         mask[r, c] = 255
    #     cv2.imshow("masked", np.array([mask, mask, mask]).transpose((1, 2, 0)) * image)
    #     cv2.waitKey(0)
    chars = np.zeros(shape=img.shape, dtype=np.uint8)
    for component in mask:
        for r, c in component:
            chars[r, c] = img[r, c]
    cv2.imwrite("original.jpg", img)
    cv2.imwrite("edge.jpg", edge)
    cv2.imwrite("chars.jpg", chars)
    cv2.imshow("image", img)
    cv2.imshow("edge", edge)
    cv2.imshow("chars", chars)
    cv2.waitKey(0)

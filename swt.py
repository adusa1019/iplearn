# coding=utf-8

import cv2
import math
import numpy as np
import scipy
import time


class stroke_width_transform():
    def __init__(self):
        pass

    def run(self, image):
        cv2.imshow("image", image)
        # 1. The Stroke Width Transform
        swt, edge = self.calculate_stroke_width_transform(image)
        cv2.imshow("swt", swt)
        # 2. Finding Letter Candidates
        connect_components = self.connect_components(swt)
        mask_components = self.find_letters(swt, connect_components)
        # for component in mask_components:
        #     mask = np.zeros(shape=swt.shape)
        #     for r, c in component:
        #         mask[r, c] = 255
        #     cv2.imshow("masked", np.array([mask, mask, mask]).transpose((1, 2, 0)) * image)
        #     cv2.waitKey(0)
        chars = np.zeros(shape=image.shape, dtype=np.uint8)
        for component in mask_components:
            for r, c in component:
                chars[r, c] = image[r, c]
        cv2.imshow("chars", chars)
        cv2.imwrite("chars.jpg", chars)
        cv2.waitKey(0)
        # 3. Grouping Letters into Text Lines
        pass

    @staticmethod
    def get_derivative(image, low=100, high=300):
        sobel_filter_size = -1
        edge = cv2.Canny(image, low, high)
        dx = cv2.Sobel(image, cv2.CV_64F, 1, 0, sobel_filter_size)
        dy = cv2.Sobel(image, cv2.CV_64F, 0, 1, sobel_filter_size)
        return edge, dx, dy, np.arctan2(dy, dx) * edge / np.max(edge)

    @staticmethod
    def is_valid_index(shape, x, y):
        return 0 <= y < shape[0] and 0 <= x < shape[1]

    def calculate_stroke_width_transform(self, image, direction=1):
        if image.ndim == 2:
            gray_image = image
        else:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # get image edge(threshold values are temporary)
        edge, dx, dy, theta = self.get_derivative(gray_image)
        edge_coordinates = [(r, c) for r, c in np.transpose(edge.nonzero())]
        ex, ey = np.cos(theta), np.sin(theta)

        # initialize SWT
        swt = np.empty(gray_image.shape)
        swt.fill(np.infty)

        # x:horizontal, y: vertical
        start = time.time()
        rays = []
        test = False
        if test:
            for y, x in edge_coordinates:
                ray = self.cast_ray(y, x, theta, edge)
                if ray:
                    rays.append(ray)
        else:
            for y, x in edge_coordinates:
                ray = []
                curr_x, curr_y = x, y
                base_ex, base_ey = ex[y][x], ey[y][x]

                ray.append([y, x])
                # print("orig theta: %f" % theta[y, x])

                # for i in range(max(image_shape) * 2):
                for i in range(1, 25):
                    next_x = math.floor(x - base_ex * i * direction)
                    next_y = math.floor(y - base_ey * i * direction)

                    # current pixel == the next pixel
                    if next_x == curr_x and next_y == curr_y:
                        continue
                    # the next pixel is image boundary
                    if not self.is_valid_index(gray_image.shape, next_x, next_y):
                        break

                    ray.append([next_y, next_x])
                    curr_x, curr_y = next_x, next_y
                    # (next_y, next_x) not in edge_coordinates
                    if edge[next_y][next_x] == 0:
                        continue
                    epsilon = 0.5
                    if np.dot([base_ey, base_ex], [ey[next_y][next_x], ex[next_y][next_x]]) > -1 + epsilon:
                        continue

                    # print("opposite theta: %f" % theta[next_y, next_x])
                    # print("length: %d" % i)
                    rays.append(ray)
                    break
        # first time
        for ray in rays:
            stroke_width = math.floor(
                np.linalg.norm(np.array([ray[0][0], ray[0][1]]) - np.array([ray[-1][0], ray[-1][1]])))
            for r_y, r_x in ray:
                swt[r_y, r_x] = min(stroke_width, swt[r_y, r_x])
        # second time
        for ray in rays:
            median = np.median([swt[r_y][r_x] for r_y, r_x in ray])
            for r_y, r_x in ray:
                swt[r_y, r_x] = min(median, swt[r_y, r_x])

        print(time.time() - start)
        # cv2.imshow("edge", edge)
        # cv2.imshow("swt", swt)
        # cv2.waitKey(0)
        # cv2.imwrite("edge.jpg", edges)
        # cv2.imwrite("swt.jpg", swt * 128)
        swt[np.isinf(swt)] = 0
        return swt, edge

    def cast_ray(self, start_x, start_y, theta, edge, max_length=50, direction=1):
        ray_direction = theta[start_y][start_x]

        ray = [[start_y, start_x]]
        for ray_length in range(1, max_length):
            next_x = math.floor(start_x + math.cos(ray_direction) * ray_length * direction)
            next_y = math.floor(start_y + math.sin(ray_direction) * ray_length * direction)
            if not self.is_valid_index(theta.shape, next_x, next_y):
                return None
            ray.append([next_y, next_x])
            if edge[next_y][next_x] == 0:
                continue
            opposite_direction = theta[next_y][next_x]
            if abs(ray_direction - opposite_direction) > math.pi / 2:
                return ray
        return None

    def connect_components(self, swt):
        start = time.time()
        swt_coordinates = np.transpose(swt.nonzero())
        label_map = np.zeros(shape=swt.shape)
        next_label = 0
        neighbors_coordinates = [[(y - 1, x - 1),  # northwest
                                  (y - 1, x),  # north
                                  (y - 1, x + 1),  # northeast
                                  (y, x - 1)]  # west
                                 for y, x in swt_coordinates]
        ratio_threshold = 3.0
        for current, neighbors in zip(swt_coordinates, neighbors_coordinates):
            next_label += 1
            label_map[current[0], current[1]] = next_label
            swt_current = swt[current[0], current[1]]
            label = next_label
            for neighbor in neighbors:
                if not self.is_valid_index(swt.shape[:2], neighbor[1], neighbor[0]):
                    continue
                if swt[neighbor] == 0:
                    continue
                swt_neighbor = swt[neighbor]
                if swt_neighbor / swt_current < ratio_threshold and swt_current / swt_neighbor < ratio_threshold:
                    label = min(label_map[neighbor], label)
                    label_map[current[0], current[1]] = label
                    label_map[neighbor] = label
        labels = dict([])
        for i in range(1, next_label):
            if len(label_map[label_map == i]) == 0:
                continue
            # labels[str(i)] = [(r, c) for r, c in np.transpose(np.nonzero(label_map == i))]
            labels[str(i)] = np.transpose(np.nonzero(label_map == i))

            # print(np.transpose(np.nonzero(label_map == i)))
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
        mask_components = []
        for label, layer in shapes.items():
            # print(layer)
            ys, xs = np.transpose(layer)
            east, west, south, north = max(xs), min(xs), max(ys), min(ys)
            width, height = east - west, south - north

            if width < 8 or height < 8:
                continue
            if width / height > 10 or height / width > 10:
                continue
            diameter = math.sqrt(width ** 2 + height ** 2)
            median_swt = np.median([swt[r, c] for r, c in layer])
            # if diameter / median_swt > 10:
            #     continue
            if width / swt.shape[1] > 0.4 or height / swt.shape[0] > 0.4:
                continue

            mask_components.append([(r, c) for r, c in layer])

            # we use log_base_2 so we can do linear distance comparison later using k-d tree
            # ie, if log2(x) - log2(y) > 1, we know that x > 2*y
            # Assumption: we've eliminated anything with median_swt == 1
            swts.append([math.log(median_swt, 2)])
            heights.append([math.log(height, 2)])
            topleft_pts.append(np.asarray([north, west]))
            widths.append(width)
            images.append(layer)
        print(time.time() - start)
        return mask_components
        # return swts, heights, widths, topleft_pts, images


if __name__ == '__main__':
    img = cv2.imread("036.jpg", 1)
    tmp = stroke_width_transform()
    tmp.run(img)


class SWTScrubber(object):
    @classmethod
    def scrub(cls, filepath):
        """
        Apply Stroke-Width Transform to image.

        :param filepath: relative or absolute filepath to source image
        :return: numpy array representing result of transform
        """
        canny, sobelx, sobely, theta = cls._create_derivative(filepath)
        swt = cls._swt(theta, canny, sobelx, sobely)
        shapes = cls._connect_components(swt)
        swts, heights, widths, topleft_pts, images = cls._find_letters(swt, shapes)
        word_images = cls._find_words(swts, heights, widths, topleft_pts, images)

        final_mask = np.zeros(swt.shape)
        for word in word_images:
            final_mask += word
        return final_mask

    @classmethod
    def _find_words(cls, swts, heights, widths, topleft_pts, images):
        # Find all shape pairs that have similar median stroke widths
        print('SWTS')
        print(swts)
        print('DONESWTS')
        swt_tree = scipy.spatial.KDTree(np.asarray(swts))
        stp = swt_tree.query_pairs(1)

        # Find all shape pairs that have similar heights
        height_tree = scipy.spatial.KDTree(np.asarray(heights))
        htp = height_tree.query_pairs(1)

        # Intersection of valid pairings
        isect = htp.intersection(stp)

        chains = []
        pairs = []
        pair_angles = []
        for pair in isect:
            left = pair[0]
            right = pair[1]
            widest = max(widths[left], widths[right])
            distance = np.linalg.norm(topleft_pts[left] - topleft_pts[right])
            if distance < widest * 3:
                delta_yx = topleft_pts[left] - topleft_pts[right]
                angle = np.arctan2(delta_yx[0], delta_yx[1])
                if angle < 0:
                    angle += np.pi

                pairs.append(pair)
                pair_angles.append(np.asarray([angle]))

        angle_tree = scipy.spatial.KDTree(np.asarray(pair_angles))
        atp = angle_tree.query_pairs(np.pi / 12)

        for pair_idx in atp:
            pair_a = pairs[pair_idx[0]]
            pair_b = pairs[pair_idx[1]]
            left_a = pair_a[0]
            right_a = pair_a[1]
            left_b = pair_b[0]
            right_b = pair_b[1]

            # @todo - this is O(n^2) or similar, extremely naive. Use a search tree.
            added = False
            for chain in chains:
                if left_a in chain:
                    chain.add(right_a)
                    added = True
                elif right_a in chain:
                    chain.add(left_a)
                    added = True
            if not added:
                chains.append(set([left_a, right_a]))
            added = False
            for chain in chains:
                if left_b in chain:
                    chain.add(right_b)
                    added = True
                elif right_b in chain:
                    chain.add(left_b)
                    added = True
            if not added:
                chains.append(set([left_b, right_b]))

        word_images = []
        for chain in [c for c in chains if len(c) > 3]:
            for idx in chain:
                word_images.append(images[idx])
                # cv2.imwrite('keeper'+ str(idx) +'.jpg', images[idx] * 255)
                # final += images[idx]

        return word_images

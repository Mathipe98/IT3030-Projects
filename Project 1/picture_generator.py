from typing import Dict, Tuple
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import scipy.misc as smp
import random
from os import listdir
from os.path import isfile, join


def create_circular_mask(h, w, center=None, radius=None):

    if center is None:
        # Use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None:
        # Use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    circle_limit = np.random.choice(
        [0.8, 0.9, 1.1, 1.2], p=[0.25, 0.25, 0.25, 0.25])
    mask = (dist_from_center <= radius +
            circle_limit) & (dist_from_center >= radius - circle_limit)
    return mask


class PictureGenerator:
    """This class will function as a gathering for methods that create randomized pictures
    of different shapes. The pictures will be represented by 2D-matrices, where a 1 in an
    index will signify a black dot (i.e. part of figure), and a 0 will be a white dot.
    """

    def __init__(self, n: int = 30, noise: float = 0.05, data_split: Tuple = (0.7, 0.2, 0.1),
                 n_pictures: int = 1000, centered: bool = True) -> None:
        assert 10 <= n <= 50, "Picture dimensions n must be 10 <= n <= 50"
        assert round(sum(data_split),
                     8) == 1, "Dataset partitions must sum to 1"
        self.n = n
        self.noise = noise
        self.centered = centered
        self.training_ratio = data_split[0]
        self.valid_ratio = data_split[1]
        self.test_ratio = data_split[2]
        self.n_pictures = n_pictures

    def generate_random_cross(self) -> np.ndarray:
        """Method that will generate an n x n array, where n is the given parameter
        from configuration.
        Array will represent a cross, where the coloured pixels are 1's, and all other
        pixels (i.e. elements in the array) are 0

        Returns:
            np.ndarray: Array containing the aforementioned cross
        """
        # Create an empty canvas with which we can work
        img = np.zeros((self.n, self.n))
        # Let's extract a random subsection of this canvas in which we create a symbol
        rand_section = random.randrange(0, 4)
        # Create another variable that randomly picks out an index of the canvas
        small_section = self.get_random_small_section(rand_section, img)
        # Now we've extracted a small section of the canvas, and we now turn the diagonals black of
        # the small sections. Then we randomize (generate noise to) the result, and return it
        np.fill_diagonal(small_section, 1)
        # A percentage of the time, we make the cross thicker
        p = [0.666667, 0.333333]
        if np.random.choice([0, 1], p=p) == 1:
            s = np.arange(len(small_section))
            small_section[s[:-1], s[1:]] = 1
            small_section.T[s[:-1], s[1:]] = 1
        small_section = np.flip(small_section, axis=1)
        # Now do the same, only for the other diagonal
        if np.random.choice([0, 1], p=p) == 1:
            s = np.arange(len(small_section))
            small_section[s[:-1], s[1:]] = 1
            small_section.T[s[:-1], s[1:]] = 1
        np.fill_diagonal(small_section, 1)
        img = self.generate_noise(img)
        return img

    def generate_random_circle(self) -> np.ndarray:
        """Same as above function, however returns a circle shape

        Returns:
            np.ndarray: Array that represents a circle picture
        """
        img = np.zeros((self.n, self.n))
        rand_section = random.randrange(0, 4)
        small_section = self.get_random_small_section(rand_section, img)
        h, w = small_section.shape
        mask = create_circular_mask(h, w)
        small_section[mask] = 1
        img = self.generate_noise(img)
        return img

    def generate_random_horizontal_lines(self) -> np.ndarray:
        """Same as above two, however generates somewhat randomized
        horizontal vertical lines.

        Returns:
            np.ndarray: Array containing representation of lines
        """
        img = np.zeros((self.n, self.n))
        rand_section = random.randrange(0, 4)
        small_section = self.get_random_small_section(rand_section, img)
        # Create a loop that continuously adds horizontal lines until we reach the end of the array
        current_index = np.random.choice([0, 1], p=[0.5, 0.5])
        while True:
            n_skip_lines = np.random.choice([4, 5, 6], p=[0.33, 0.33, 0.34])
            if current_index + n_skip_lines >= small_section.shape[0] - 1:
                break
            small_section[current_index, :] = 1
            current_index += n_skip_lines
        img = self.generate_noise(img)
        return img

    def generate_random_vertical_lines(self) -> np.ndarray:
        """Same as the horizontal method, only transposed

        Returns:
            np.ndarray: Array containing randomized vertical lines
        """
        return self.generate_random_horizontal_lines().T

    def get_random_small_section(self, rand_section: int, img: np.ndarray) -> np.ndarray:
        """This method will return a small subsection of a larger 2D array (which represents
        a picture) which is used to generate randomized shapes in various locations.

        Args:
            rand_section (int): Integer describing which section of the array we'll use.
                        Mapping: 0 => top-left; 1 => top-right; 2 => bottom-left; 3 => bottom-right
            img (np.ndarray): The array of the original image

        Returns:
            np.ndarray: The small subsection cut-out of the input
        """
        # If we want centered objects, then we just return the image itself
        if self.centered:
            return img
        random_index = random.randrange(
            int(np.ceil(self.n * 0.3)), int(np.ceil(self.n * 0.8)))
        if rand_section == 0:
            small_section = img[:random_index, :random_index]
        elif rand_section == 1:
            small_section = img[:random_index, self.n - random_index:]
        elif rand_section == 2:
            small_section = img[self.n - random_index:, :random_index]
        else:
            small_section = img[self.n - random_index:, self.n - random_index:]
        return small_section

    def generate_noise(self, img: np.ndarray) -> np.ndarray:
        # Create a random variable that will determine if we randomize one pixel at a time
        # The probability is taken from the noise parameter
        p = [1-self.noise * 10, self.noise * 10]
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                current_pixel_value = img[i][j]
                # If the current pixel is black, then we might randomly make it white and make another
                # white pixel become black
                if current_pixel_value == 1:
                    if np.random.choice([0, 1], p=p) == 1:
                        r_i, r_j = random.randrange(
                            0, self.n), random.randrange(0, self.n)
                        img[i][j] = 0
                        img[r_i][r_j] = 1
                # For each pixel, we might also randomly make it black (if it isn't already)
                else:
                    p2 = [1-self.noise, self.noise]
                    if np.random.choice([0, 1], p=p2) == 1:
                        img[i][j] = 1
        return img

    def generate_datasets(self) -> None:
        """Method for generating all the necessary datasets.
        Takes a while to run, so it will just be run once to save all files,
        such that the files can be read from memory rather than created on the fly.

        Note: was run once for centered images, and another time for non-centered images.
        All images are 50x50 (but this can be changed by the input n)
        """
        for i in range(self.n_pictures):
            cross = self.generate_random_cross()
            circ = self.generate_random_circle()
            h_line = self.generate_random_horizontal_lines()
            v_line = self.generate_random_vertical_lines()
            cross_path = f"./datasets/crosses_non_centered/cross_{i+1}.png"
            circle_path = f"./datasets/circles_non_centered/circle_{i+1}.png"
            h_line_path = f"./datasets/h_lines_non_centered/h_line_{i+1}.png"
            v_line_path = f"./datasets/v_lines_non_centered/v_line_{i+1}.png"
            self.save_img(cross, cross_path)
            self.save_img(circ, circle_path)
            self.save_img(h_line, h_line_path)
            self.save_img(v_line, v_line_path)
            if i % 50 == 0:
                print(i)

    def show_img(self, img: np.ndarray) -> None:
        # Creates PIL image
        img = Image.fromarray(np.uint8(255 - img * 255), 'L')
        img = img.resize((500, 500), Image.NEAREST)
        img.show()

    def save_img(self, img: np.ndarray, path: str) -> None:
        img = Image.fromarray(np.uint8(255 - img * 255), 'L')
        # img = img.resize((500, 500), Image.NEAREST)
        img.save(path)

    def get_datasets(self) -> Dict:
        crosses_path = './datasets/crosses_centered' if self.centered else './datasets/crosses_non_centered'
        crosses = [f for f in listdir(
            crosses_path) if isfile(join(crosses_path, f))]
        circles_path = './datasets/circles_centered' if self.centered else './datasets/circles_non_centered'
        circles = [f for f in listdir(
            circles_path) if isfile(join(circles_path, f))]
        h_lines_path = './datasets/h_lines_centered' if self.centered else './datasets/h_lines_non_centered'
        h_lines = [f for f in listdir(
            h_lines_path) if isfile(join(h_lines_path, f))]
        v_lines_path = './datasets/v_lines_centered' if self.centered else './datasets/v_lines_non_centered'
        v_lines = [f for f in listdir(
            v_lines_path) if isfile(join(v_lines_path, f))]
        num_training_examples = int(
            np.ceil(self.training_ratio * len(crosses)))
        num_valid_examples = int(np.ceil(self.valid_ratio * len(crosses)))
        num_test_examples = int(np.ceil(self.test_ratio * len(crosses)))
        shape_solutions = {
            "cross": [1, 0, 0, 0],
            "circle": [0, 1, 0, 0],
            "h_line": [0, 0, 1, 0],
            "v_line": [0, 0, 0, 1]
        }
        datasets = {
            "training": [],
            "training_targets": [],
            "validation": [],
            "validation_targets": [],
            "testing": [],
            "testing_targets": []
        }
        for i in range(num_training_examples + num_valid_examples
                       + num_test_examples):
            if i < num_training_examples:
                dataset = "training"
            elif i < num_training_examples + num_valid_examples:
                dataset = "validation"
            else:
                dataset = "testing"
            cross = crosses[i]
            circle = circles[i]
            h_line = h_lines[i]
            v_line = v_lines[i]
            cr_fp = f"{crosses_path}/{cross}"
            cir_fp = f"{circles_path}/{circle}"
            hl_fp = f"{h_lines_path}/{h_line}"
            vl_fp = f"{v_lines_path}/{v_line}"
            datasets[dataset].append(
                np.where(np.array(Image.open(cr_fp)) == 0, 1, 0))
            datasets[dataset].append(
                np.where(np.array(Image.open(cir_fp)) == 0, 1, 0))
            datasets[dataset].append(
                np.where(np.array(Image.open(hl_fp)) == 0, 1, 0))
            datasets[dataset].append(
                np.where(np.array(Image.open(vl_fp)) == 0, 1, 0))
            for _, shape_solution in shape_solutions.items():
                datasets[f"{dataset}_targets"].append(shape_solution)
        return datasets
        


def generate_new_dataset() -> None:
    params = {
        "n": 50,
        "noise": 0.02,
        "centered": False,
    }
    generator = PictureGenerator(**params)
    generator.generate_datasets()


if __name__ == '__main__':
    params = {
        "n": 50,
        "noise": 0.02,
        "centered": False,
    }
    pg = PictureGenerator(**params)
    data = pg.get_datasets()
    for key, value in data.items():
        print(f"{key}:\n {value[0]}")

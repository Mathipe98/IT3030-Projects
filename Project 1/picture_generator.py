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
    # Create a randomized bound to vary the thickness of the circle
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

    def __init__(self, n: int = 30, noise: float = 0.05,
                 data_split: Tuple = (0.7, 0.2, 0.1),
                 centered: bool = True, n_pictures: int = 100,
                 generate_realtime: bool = False, verbose: bool = True,
                 flatten: bool=False) -> None:
        """Constructor for necessary parameters for picture generation.

        Args:
            n (int, optional): Size of the randomized picture. Defaults to 30.
            noise (float, optional): Percentage of randomization in the picture. Defaults to 0.05.
            data_split (Tuple, optional): The split between training, validation, and testing. Defaults to (0.7, 0.2, 0.1).
            centered (bool, optional): Whether or not pictures should be centered. Defaults to True.
            n_pictures (int, optional): Number of pictures to generate/extract in TOTAL. Defaults to 100.
            generate_realtime (bool, optional): Whether or not to generate pictures realtime, or fetch from local directory. Defaults to False.
            verbose (bool, optional): Whether or not to include additional print statements. Defaults to True.
            flatten (bool, optional): Whether or not to flatten image arrays. Defaults to False.
        """
        assert round(sum(data_split),8) == 1, "Dataset partitions must sum to 1"
        self.n = n
        self.noise = noise
        self.centered = centered
        self.training_ratio = data_split[0]
        self.valid_ratio = data_split[1]
        self.test_ratio = data_split[2]
        self.n_pictures = n_pictures
        self.generate_realtime = generate_realtime
        self.verbose = verbose
        self.flatten = flatten

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
        """This method takes in a 2D image and generates noise in it according
        to the noise-parameter from the config file.

        Args:
            img (np.ndarray): 2D array of pixel values

        Returns:
            np.ndarray: Slightly altered input array with more noise/randomization/entropy
        """
        # Create a random variable that will determine if we randomize one pixel at a time
        # The probability is taken from the noise parameter
        p = [1-self.noise, self.noise]
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

    def write_datasets(self, path: str) -> None:
        """Method that generates and writes randomized pictures of shapes to
        a local directory

        Args:
            path (str): Folder name inside the 'datasets' directory in which to store files
        """
        add = "_non" if not self.centered else ""
        for i in range(self.n_pictures):
            cross = self.generate_random_cross()
            circ = self.generate_random_circle()
            h_line = self.generate_random_horizontal_lines()
            v_line = self.generate_random_vertical_lines()
            cross_path = f"./datasets/{path}/crosses{add}_centered/cross_{i+1}.png"
            circle_path = f"./datasets/{path}/circles{add}_centered/circle_{i+1}.png"
            h_line_path = f"./datasets/{path}/h_lines{add}_centered/h_line_{i+1}.png"
            v_line_path = f"./datasets/{path}/v_lines{add}_centered/v_line_{i+1}.png"
            self.save_img(cross, cross_path)
            self.save_img(circ, circle_path)
            self.save_img(h_line, h_line_path)
            self.save_img(v_line, v_line_path)
            if (i+1) % self.n_pictures // 10 == 0 and self.verbose:
                print(f"Writing datasets.. {(i+1)*10}% complete")
        print(f"Writing datasets.. 100% complete")

    def get_datasets(self) -> Dict:
        """Generic function for retrieving datasets.
        Only delegates further down the chain, but checks whether or not
        to fetch from files, or generate on the fly

        Returns:
            Dict: Dictionary containing the necessary datasets
        """
        if self.generate_realtime:
            if self.verbose:
                print("Fetching datasets realtime...")
            return self.get_datasets_realtime()
        if self.verbose:
            print("Fetching datasets from directory...")
        return self.get_datasets_from_directory()

    def get_datasets_realtime(self) -> Dict:
        """Method that generates datasets during runtime

        Returns:
            Dict: Dictionary containing the necessary datasets
        """
        datasets = {
            "training": [],
            "training_targets": [],
            "validation": [],
            "validation_targets": [],
            "testing": [],
            "testing_targets": []
        }
        shape_solutions = {
            "cross": np.array([1, 0, 0, 0]).reshape(4, 1),
            "circle": np.array([0, 1, 0, 0]).reshape(4, 1),
            "h_line": np.array([0, 0, 1, 0]).reshape(4, 1),
            "v_line": np.array([0, 0, 0, 1]).reshape(4, 1)
        }
        num_training_examples = int(
            np.ceil(self.training_ratio * self.n_pictures) // 4)
        num_valid_examples = int(np.ceil(self.valid_ratio * self.n_pictures) // 4)
        num_test_examples = int(np.ceil(self.test_ratio * self.n_pictures) // 4)
        if self.verbose:
            print(f"N train examples:\t {num_training_examples*4}\nN valid examples:\t {num_valid_examples*4}\nN test examples:\t {(num_test_examples+1)*4}")
        for i in range(num_training_examples + num_valid_examples
                       + num_test_examples + 1):
            if i <= num_training_examples - 1:
                dataset = "training"
            elif i <= (num_training_examples + num_valid_examples) - 1:
                dataset = "validation"
            else:
                dataset = "testing"
            cross = self.generate_random_cross()
            circ = self.generate_random_circle()
            h_line = self.generate_random_horizontal_lines()
            v_line = self.generate_random_vertical_lines()
            if self.flatten:
                datasets[dataset].append(
                    np.where(cross == 0, 1, 0).reshape(self.n ** 2, 1))
                datasets[dataset].append(
                    np.where(circ == 0, 1, 0).reshape(self.n ** 2, 1))
                datasets[dataset].append(
                    np.where(h_line == 0, 1, 0).reshape(self.n ** 2, 1))
                datasets[dataset].append(
                    np.where(v_line == 0, 1, 0).reshape(self.n ** 2, 1))
            else:
                datasets[dataset].append(
                    np.where(cross == 0, 1, 0))
                datasets[dataset].append(
                    np.where(circ == 0, 1, 0))
                datasets[dataset].append(
                    np.where(h_line == 0, 1, 0))
                datasets[dataset].append(
                    np.where(v_line == 0, 1, 0))
            for _, shape_solution in shape_solutions.items():
                datasets[f"{dataset}_targets"].append(shape_solution)
        return datasets

    def get_datasets_from_directory(self) -> Dict:
        """Method that retrieves the necessary datasets from memory
        (file storage), rather than producing during code-execution.

        Returns:
            Dict: Dictionary containing the necessary datasets
        """
        datasets = {
            "training": [],
            "training_targets": [],
            "validation": [],
            "validation_targets": [],
            "testing": [],
            "testing_targets": []
        }
        shape_solutions = {
            "cross": np.array([1, 0, 0, 0]).reshape(4, 1),
            "circle": np.array([0, 1, 0, 0]).reshape(4, 1),
            "h_line": np.array([0, 0, 1, 0]).reshape(4, 1),
            "v_line": np.array([0, 0, 0, 1]).reshape(4, 1)
        }
        crosses_path = './datasets/pregenerated/crosses_centered' if self.centered else './datasets/pregenerated/crosses_non_centered'
        crosses = [f for f in listdir(
            crosses_path) if isfile(join(crosses_path, f))]
        circles_path = './datasets/pregenerated/circles_centered' if self.centered else './datasets/pregenerated/circles_non_centered'
        circles = [f for f in listdir(
            circles_path) if isfile(join(circles_path, f))]
        h_lines_path = './datasets/pregenerated/h_lines_centered' if self.centered else './datasets/pregenerated/h_lines_non_centered'
        h_lines = [f for f in listdir(
            h_lines_path) if isfile(join(h_lines_path, f))]
        v_lines_path = './datasets/pregenerated/v_lines_centered' if self.centered else './datasets/pregenerated/v_lines_non_centered'
        v_lines = [f for f in listdir(
            v_lines_path) if isfile(join(v_lines_path, f))]
        num_training_examples = int(
            np.ceil(self.training_ratio * self.n_pictures) // 4)
        num_valid_examples = int(np.ceil(self.valid_ratio * self.n_pictures) // 4)
        num_test_examples = int(np.ceil(self.test_ratio * self.n_pictures) // 4)
        if self.verbose:
            print(f"N train examples:\t {num_training_examples*4}\nN valid examples:\t {num_valid_examples*4}\nN test examples:\t {(num_test_examples+1)*4}")
        for i in range(num_training_examples + num_valid_examples
                       + num_test_examples + 1):
            if i <= num_training_examples - 1:
                dataset = "training"
            elif i <= (num_training_examples + num_valid_examples) - 1:
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
                np.where(np.array(Image.open(cr_fp)) == 0, 1, 0).reshape(self.n ** 2, 1))
            datasets[dataset].append(
                np.where(np.array(Image.open(cir_fp)) == 0, 1, 0).reshape(self.n ** 2, 1))
            datasets[dataset].append(
                np.where(np.array(Image.open(hl_fp)) == 0, 1, 0).reshape(self.n ** 2, 1))
            datasets[dataset].append(
                np.where(np.array(Image.open(vl_fp)) == 0, 1, 0).reshape(self.n ** 2, 1))
            for _, shape_solution in shape_solutions.items():
                datasets[f"{dataset}_targets"].append(shape_solution)
        return datasets

    def show_img(self, img: np.ndarray) -> None:
        """Method that upscales and shows an arbitrary 2D matrix image

        Args:
            img (np.ndarray): 2D matrix that one wants to visualize
        """
        # Creates PIL image
        img = Image.fromarray(np.uint8(img * 255), 'L')
        img = img.resize((300, 300), Image.NEAREST)
        img.show()

    def save_img(self, img: np.ndarray, path: str) -> None:
        """Method that saves a given image in the form of a 2D
        matrix to a provided path

        Args:
            img (np.ndarray): 2D matrix in question
            path (str): Local path in which to save the image
        """
        img = Image.fromarray(np.uint8(img * 255), 'L')
        # img = img.resize((500, 500), Image.NEAREST)
        img.save(path)

if __name__ == "__main__":
    params = {
        "n": 50,
        "noise": 0.5,
        "data_split": (0.6, 0.3, 0.1),
        "centered": True,
        "n_pictures": 100,
        "generate_realtime": True,
        "verbose": True,
        "flatten": False
    }
    n = params["n"]
    generator = PictureGenerator(**params)
    datasets = generator.get_datasets()
    example1 = datasets["training"][0]
    example2 = datasets["training"][1]
    example3 = datasets["training"][2]
    example4 = datasets["training"][3]
    # print(example1)
    generator.show_img(example1)
    generator.show_img(example2)
    generator.show_img(example3)
    generator.show_img(example4)
    
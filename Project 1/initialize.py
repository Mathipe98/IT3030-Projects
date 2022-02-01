from typing import Dict
import numpy as np
from network_structures import  NeuralNetwork
from nn_functions import *
from visualization import draw_network
from picture_generator import PictureGenerator
import configparser
import ast

# Create a dict so we can map string functions from config to Callables
function_map = {
    "relu": relu,
    "sigmoid": sigmoid,
    "tanh": tanh,
    "identity": identity,
    "softmax": softmax,
    "cross_entropy": cross_entropy,
    "mse": mse,
}

def parse_config() -> Dict:
    config = configparser.ConfigParser()
    config.read('config.txt')
    config_dict = {}
    for section in config.sections():
        if section == "IMMUTABLE":
            config_dict["inputs"] = int(config.get(section, "inputs"))
            config_dict["outputs"] = int(config.get(section, "outputs"))
        elif section == "GLOBALS":
            config_dict["lr"] = float(config.get(section, "lr"))
            config_dict["wreg"] = config.get(section, "wreg")
            config_dict["wreg_lr"] = float(config.get(section, "wreg_lr"))
            config_dict["l_func"] = function_map[config.get(section, "l_func")]
            config_dict["softmax"] = bool(config.get(section, "softmax"))
        elif section == "LAYERS":
            config_dict["n_hl"] = int(config.get(section, "n_hl"))
            config_dict["hl_neurons"] = ast.literal_eval(config.get(section, "hl_neurons"))
            hl_funcs = ast.literal_eval(config.get(section, "hl_funcs"))
            config_dict["hl_funcs"] = [function_map[func_name] for func_name in hl_funcs]
            config_dict["output_func"] = function_map[config.get(section, "output_func")]
    return config_dict


def start() -> None:
    config = parse_config()
    network = NeuralNetwork(**config)
    pg_params = {
        "n": 50,
        "centered": False,
    }
    pg = PictureGenerator(**pg_params)
    dataset = pg.get_datasets()
    print("Dataset fetched.")
    training_data = dataset["training"][0:300]
    training_targets = dataset["training_targets"][0:300]
    print(f"Starting training with {len(training_data)} training examples.")
    network.train(training_data, training_targets)

    test_example = dataset["training"][220]
    test_solution = dataset["training_targets"][220]
    prediction = network.predict(test_example)
    print(f"Final network prediction:\n \tTarget: {test_solution}\n \tPrediction: {prediction}\n \tLoss: {network.l_func(prediction, test_solution, sigmoid=True)}")

    network.visualize_training_losses()


# LAST FEW TWEAKS:
#   - Add picture parameters to config file
#   - Tweak picture generator to create the background canvas
#   - Somehow squish the 2D pictures into 1D arrays of arbitrary length
#   - Visualize validation loss
#   - Print/visualize testing ACCURACY (i.e. take the highest output of the network and make this value 1, and compare it to target)
#   - THEN YOU'RE FUCKIN DONE!


if __name__ == '__main__':
    start()
from typing import Dict
from network_structures import  NeuralNetwork
from nn_functions import *
from picture_generator import PictureGenerator
import configparser
import ast
import warnings

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
    """Function that will parse a local configuration file in order to easily setup
    different parameters for the neural network.

    Returns:
        Dict: Dictionary containing all the parameters necessary for the neural network
    """
    config = configparser.ConfigParser()
    config.read('config.txt')
    network_config = {}
    pic_gen_config = {}
    for section in config.sections():
        if section == "IMMUTABLE":
            network_config["outputs"] = int(config.get(section, "outputs"))
        elif section == "GLOBALS":
            network_config["inputs"] = int(config.get(section, "inputs"))
            network_config["lr"] = float(config.get(section, "lr"))
            network_config["wreg"] = config.get(section, "wreg")
            network_config["wreg_lr"] = float(config.get(section, "wreg_lr"))
            network_config["epochs"] = int(config.get(section, "epochs"))
            network_config["l_func"] = function_map[config.get(section, "l_func")]
            network_config["use_softmax"] = ast.literal_eval(config.get(section, "use_softmax"))
            network_config["verbose"] = ast.literal_eval(config.get(section, "verbose"))
        elif section == "LAYERS":
            network_config["n_hl"] = int(config.get(section, "n_hl"))
            network_config["hl_neurons"] = ast.literal_eval(config.get(section, "hl_neurons"))
            hl_funcs = ast.literal_eval(config.get(section, "hl_funcs"))
            network_config["hl_funcs"] = [function_map[func_name] for func_name in hl_funcs]
            network_config["hl_wranges"] = ast.literal_eval(config.get(section, "hl_wranges"))
            network_config["output_func"] = function_map[config.get(section, "output_func")]
            network_config["output_wrange"] = ast.literal_eval(config.get(section, "output_wrange"))
        elif section == "DATA":
            pic_gen_config["n"] = int(config.get(section, "n"))
            pic_gen_config["noise"] = float(config.get(section, "noise"))
            pic_gen_config["data_split"] = ast.literal_eval(config.get(section, "data_split"))
            pic_gen_config["centered"] = ast.literal_eval(config.get(section, "centered"))
            pic_gen_config["n_pictures"] = int(config.get(section, "n_pictures"))
            pic_gen_config["generate_realtime"] = ast.literal_eval(config.get(section, "generate_realtime"))
            pic_gen_config["verbose"] = ast.literal_eval(config.get(section, "verbose"))
    assert round(sum(pic_gen_config["data_split"]),8) == 1, "Dataset partitions must sum to 1"
    assert network_config["inputs"] == pic_gen_config["n"] ** 2, "Input dimensions must equal n^2"
    if not pic_gen_config["generate_realtime"]:
        warnings.warn("Parameters (inputs, n, noise) have no effect when reading from directory (this is due to their fixed size)")
        pic_gen_config["n"] = 30
        network_config["inputs"] = 30 ** 2
    return network_config, pic_gen_config


def start() -> None:
    """Function that will 'fire up' the neural network by extracting network parameters
    and all the necessary data from the picture-generator.
    """
    network_config, pg_config = parse_config()
    network = NeuralNetwork(**network_config)
    pg = PictureGenerator(**pg_config)
    dataset = pg.get_datasets()
    print("Dataset fetched.")
    training_data = dataset["training"]
    training_targets = dataset["training_targets"]
    validation_data = dataset["validation"]
    validation_targets = dataset["validation_targets"]
    testing_data = dataset["testing"]
    testing_targets = dataset["testing_targets"]
    print(f"Starting training with {len(training_data)} training examples, {len(validation_data)} validation examples, and {len(testing_data)} testing examples.")
    network.train(training_data, training_targets, validation_data, validation_targets)
    network.calculate_testing_loss(testing_data, testing_targets)
    network.visualize_losses()
    network.visualize_testing_accuracy(testing_data, testing_targets)

if __name__ == '__main__':
    start()

import numpy as np
from network_structures import *
from nn_functions import *
from visualization import DrawNN
from picture_generator import PictureGenerator
from timeit import default_timer as timer

config = {
        "Inputs": 2500,
        "Outputs": 4,
        "Hidden layers": 2,
        "HL Activation functions": [relu, relu],
        "HL Neurons": [100, 20],
        "Output function": softmax,
        "Loss function": mse,
        "Learning rate": 0.01,
        "Batch size": 1,
    }


# User testing functions

def draw_network() -> None:
    obj = NeuralNetwork(config)
    nodes = [config['Inputs']]
    for layer in obj.hidden_layers:
        nodes.append(layer.n_nodes)
    nodes.append(obj.output_layer.n_nodes)
    network_vis = DrawNN(nodes)
    network_vis.draw()

def test_training() -> None:
    network = NeuralNetwork(config)
    pg_params = {
        "n": 50,
        "centered": False,
    }
    pg = PictureGenerator(**pg_params)
    dataset = pg.get_datasets()
    print("Dataset fetched.")
    training_data = dataset["training"]
    training_targets = dataset["training_targets"]
    print(f"Starting training with {len(training_data)} training examples.")
    network.train(training_data, training_targets)

    test_example = dataset["training"][502]
    test_solution = dataset["training_targets"][502]
    prediction = network.predict(test_example)
    print(f"Final network prediction:\n \tTarget: {test_solution}\n \tPrediction: {prediction}\n \tLoss: {network.l_func(prediction, test_solution, sigmoid=True)}")

    network.visualize_training_losses()


    


if __name__ == '__main__':
    test_training()
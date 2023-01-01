# IT3030-Projects

This is a repository containing code for the course IT3030 - Deep Learning, taken at NTNU in the spring of 2022.
The course consisted of 3 main projects, all of which were dealing with exploring the details of modern AI models and solutions,
and getting a good sense for the mathematical background, as well as the general application of models.

## Project 1

This project consisted of creating a neural network from scratch in an optimized manner. The point was to create modular code where the
various components of a neural network were clearly separated, and the mathematical operations were optimized and easily accessible and transparent.
The final result was highly optimized with use of NumPy, and most often using the latest and quickest methods available - especially Einsum.

The evaluation and testing of the model was done by generating figures consisting of 4 different figures:

* A cross
* A circle
* A horizontal line
* A vertical line

The model was then trained to classify images it was fed, where there were different versions of the data - for example figures with and without noise, as well as centered vs. non-centered figures.

Evaluating and looking at the model first-hand can be done by simply running the `initialize.py`-file.

## Project 2

The second project dealt with looking at AutoEncoders, i.e. generative AI-models that use an encoder-decoder structure in order for the former to learn the structure of the input data and
"compress" it (or transform it) into some mathematical representation that the decoder can learn from in order to generate output based on the encoded space.

This was then advanced further to use a variational autoencoder, in which the encoder is taught that the correct encoding is a specific probability distribution. Since the decoder is then taught
to use this encoding to generate "real" data, the architecture becomes better for general generation - because in order to generate any data, you can simply sample from the same probability distribution
that the encoder is taught to approximate, and feed it directly into the decoder.

The testing and evaluation was based on MNIST-data, mainly pictures of hand-written digits. Both single digits, as well as triple-stacked digits were used. The idea was to get to a point where the architecture
could generate data that didn't seem much different from the real data, and visualizations of the results can be seen inside the 'figures'-folder.

In order to run and visualize this, one can simply run the `variational_autoencoder.py`-file. The general results 

## Project 3

The final project concerned LSTM models, and usage of them in problems dealing with time and sequential data. The task was to predict a certain metric of the power-grid in a region of Norway.
It was a simple timeseries-prediction task, however certain caveats had to be taken into consideration. For example - an important metric for a timeseries, is the target variable one timestep prior.
Thus during prediction, this metric must be generated on-the-fly such that the model can use it for the next prediction. However one must also be careful to avoid leaking data into the prediction,
i.e. indirectly disclosing the answer.

Small challenges like this, as well as the general theory behind LSTM models and handling "memory" in AI, was the main focus of the project.

The evaluation and testing of the model was done by simply measuring the accuracy of the predicted data compared to the answer.

The `start.ipynb`-file provides everything needed to run and test the model.
# Violence Detection using Dense Multi Head Self-Attention with Bidirectional Convolutional LSTM


This is the code for the paper

[ViolenceNet: Dense Multi Head Self-Attention with Bidirectional Convolutional LSTM for Detecting Violence<br/>
Fenando José Rendón Segador, Juan Antonio Álvarez-García, Fernando Enriquez, Oscar Deniz](https://www.mdpi.com/2079-9292/10/13/1601/htm)

The paper addresses the problem of detecting violent actions in videos, analyzing the state of the art of various deep learning models and developing a new one. Our aim is to develop a new neural network model capable of reaching or exceeding the state of the art.

## Requirements

This project is implemented in Python using the [Tensorflow](https://www.tensorflow.org/) and [Keras](https://keras.io/) libraries to develop the model.

- Tensorflow v2.5.0
- Sci-Kit Learn

## Model Architecture

The following graph shows the architecture of our model.

![Model Architecture](figures/ModelArchitecture.png?raw=True "Model Architecture")

In section A the architecture of the ViolenceNet model, that takes the optical flow as input, is shown. It is composed of four parts: a DenseNet-121 network spatio-temporal encoder, a multi-head self-attention layer, a bidirectional convolution 2D LSTM (BiConvLSTM2D) layer and a classifier. Below each Dense Block its number of components is indicated. The variable h corresponds to the number of heads used in parallel by the multi-head self-attention layer and the variables Q,K,V their inputs. Section B shows the internal architecture of a 5-component DenseBlock (x5).

## Results

## Evaluation

We provide a [Jupyter Notebook](ViolenceActionDetection.ipynb) with instructions to train and test our model.

In order to run the application it is necessary to download the datasets. And to download the datasets it is necessary to upload the kaggle.json file found in this repository.

The following steps must be followed:

1. First step

    Download the kaggle.json file to a local environment

2. Second step

    Open the jupyter notebook [Jupyter Notebook](ViolenceActionDetection.ipynb)

3. Third step

    Execute the first cell of the jupyter notebook.

4. Step four

    After executing the first cell, it will ask you to load the kaggle.json file. Do it.

5. Fifth step

    After the datasets is downloaded, run the other parts of the code.


## Acknowledgements
This research is partially supported by The Spanish Ministry of Economy and Competitiveness through the project VICTORY (grant no.: TIN2017-82113-C2-1-R).

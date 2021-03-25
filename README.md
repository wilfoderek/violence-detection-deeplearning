# Violence Detection using Deep Learning

This is the code for the paper

Violence detection using deep learning<br/>
Fenando José Rendón Segador, Juan Antonio Álvarez-García, Fernando Enriquez, Oscar Deniz

The article addresses the problem of detecting violent actions in videos, analyzing the state of the art of various deep learning models and developing a new one. Our aim is to develop a new neural network model capable of reaching or exceeding the state of the art.

## Requirements

This project is implemented in Python using the [Tensorflow](https://www.tensorflow.org/) and [Keras](https://keras.io/) libraries to develop the model.

## Model Architecture

The following graph shows the architecture of our model.

![Model Architecture](figures/ModelArchitecture.png?raw=True "Model Architecture")

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

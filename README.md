# simple-language-models

Name: Trang Dang
<br> Course: CS301 - Intro to Data Science

The experiment I have when building language model(s) using JAX and Flax. In JAX, I implemented a Multi-Layered Perceptron (MLP) model (Part 1) and a Bigram Language Model (Part 2a). In Flax, I built an MLP language model as well (Part 2b).

A language model is a statistical model that is used to predict the probability of a sequence of words occurring in a given language. It is trained on a large corpus of text and is used to generate new text or to evaluate the probability of a given sequence of words. A simple language model, such as the Bigram Language Model I built in JAX, predicts the probability of the next character in a name based only on the previous characters. The MLP language model I built in Flax is more complex, using multiple layers of neurons to predict the probability of the next character in a name based on the context of the entire names data set. While the author uses the micrograd library to implement MLP, I replicate it using the gradient facilities provided by JAX.

Here are some general steps to train a language model that I learned via this assignment:
* Preprocess and clean the text.
* Convert the text into numerical format.
* Define the model architecture: choose the type of neural network and the hyperparameters such as the number of layers, hidden units, and activation functions.
* Define the loss function: max-margin, negative log-likelihood, cross-entropy loss, RMSE,..
* Train the model: initialize the model parameters, define the optimizer, and perform the training loop for a certain number of epochs to minimize loss function.
* Update parameters: adjust the hyperparameters or the model based on gradients of loss function.
* Evaluate the model.
* Generate new text.

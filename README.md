DigitGuesser 1.0.0
==================

A small program that uses a pre-trained neural network to estimate (in real time) numbers drawn on a canvas. This was mainly an excuse to use the neural network the  [Make Your Own Neural Network book](https://www.amazon.com/Make-Your-Own-Neural-Network-ebook/dp/B01EER4Z4G) helps you build. I do recommend checking out the book, it's a great introduction to the concepts of neural networks in a gentle way (no maths or programming experience required).

![Animated screenshot showing estimation of digits based on painting](Screenshots/preview.gif?raw=true) 

Prerequisites
-------------

- python 3.6+

Python modules

- PyQt5
- Pillow
- numpy 

All are available through pip.

Usage
-----

Run DigitGuesser.py. Left click in the white square to draw, right click to clear. Estimates auto update as you draw.

If you want to play with the neural network that works behind the scenes, the scripts reside in neural_net/own/. Each script serves a single function:

1. get_data.py downloaded the MNIST data for the network to be trained on
2. neural_network.py has the main neural network class in it (this file has no main to execute)
3. train.py instantiates and trains a new instance of the neural network class on the training data. This is most likely the file you want to play with. It saves the final weights in neural_network.npz
4. test.py tests the weights in neural_network.npz against the test data, and gives an accuracy percentage as a result

License
-------

DigitGuesser is copyrighted free software made available under the terms of the GPLv3

Copyright: (C) 2017 by Keith Offer. All Rights Reserved.

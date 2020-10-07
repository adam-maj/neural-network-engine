# neural-network-engine
This is an engine I created that you can use to make neural networks. It functions similary to a library like 
TensorFlow/Keras or PyTorch where you can customize the layers, neurons, and activations of your neural network,
then pass train the neural network by passing in a dataset.

**The neural network implementation itself uses no 3rd party libraries** (and numpy is used once in the activation function for its exponentiation)

The engine itself is in NeuralNetworkEngine.py. If you wan't to read my explanation of what each part of the code is doing
you can look in the NeuralNetworkEngine.ipynb file, or if you want to see the same thing but be able to run it yourself,
you can visit the corresponding [Google Colab Notebook](https://colab.research.google.com/drive/1lOPb-V4UYwZnURc1HNYiKn4cuzKdpRTx?usp=sharing)

I used the engine to train a neural network and make predictions to demonstrate its capability. You can see this code in
the NNEBankAnalysis.py file or view the corresponding explanation in the NNEBankAnalysis.ipnb file. To view a live demo of the code
and run it yourself, you can visit the corresponding [Google Colab Notebook](https://colab.research.google.com/drive/1FWyLbDmi415bUFFmRhVFbLogtswRoHkd?usp=sharing)

## About This Project
I created this project in order to fully grasp the details of how a neural network works, including the linear algebra and calculus.
The actual calculus of backpropagation is implemented in the "predict" method of the NeuralNetwork class. I also created the activation 
functions for each neuron from scratch.

I decided to use no 3rd party libraries in numpy so I couldn't lean on them as a crutch to aid my understanding. I decided to use numpy
because its exponentiation function is far faster and more flexible then the exponentiation function that comes in the default python
math module, but that is the only case where I made an exception since it didn't inhibit my understanding (and as far as I saw it, there was
no reason to create a basic math function from scratch).
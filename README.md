# Variational Recurrent Neural Network

Variational Recurrent Neural Network implemented by Tensorflow

Last updated: Feb. 28th, 2018.
Author: Tatsuro Yamada <<ymt2.casino@gmail.com>>

## Requirements
- Python 2.7 (NO supports for 3.4 nor 3.5)
- Tensorflow 1.4
- NumPy 1.11

## Implementation
- The model used in Denton and Fergus, "Stochastic Video Generation with a Learned Prior" <https://sites.google.com/view/svglp/>

## Example
- $ cd train
- $ python learn.py
-   (It may take several minutes to download moving MNIST dataset for the first time)

### Following must be taken into consideration
- Reconstruction error (mse? cross entropy?)
- Beta (the balance between reconstruction and regularization)
- Encoder and decoder (end-to-end? importing pretrained model?)
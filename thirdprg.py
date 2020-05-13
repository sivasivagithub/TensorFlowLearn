import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

print(tf.__version__)

fashion_mnist =     keras.datasets.fashion_mnist #The Fashion MNIST data is available directly in the tf.keras datasets API. You load it like this:

(training_images, training_labels), (test_images, test_labels) = fashion_mnist.load_data()
#Calling load_data on this object will give you two sets of two lists, these will be the training and testing values for the graphics that contain the clothing items
# and their labels.


plt.imshow(training_images[0])
print(training_labels[0])
print(training_images[0])
# ====================load MNIST dataset in keras==================
# import mnist dataset
from keras.datasets import mnist

# The model will learn from the train_images and the train_labels
# The model will be tested based on test_images and test_labels
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# number of axes of the tensor "train_images"
print(train_images.ndim)
# shape of the tensor "train_images"
print(train_images.shape)
# data type of the tensor "train_images"
print(train_images.dtype)

# ======================================display data=======================================
digit = test_images[1]
import matplotlib.pyplot as plt

plt.imshow(digit, cmap=plt.cm.binary)
plt.show()

# =============== tensor slicing (selecting specific elements of a tensor)================
my_slice = train_images[10:100]
# equivalent slicing examples
"""my_slice = train_images[10:100, :, :]
my_slice = train_images[10:100, 0:28, 0:28]"""
print(my_slice.shape)

# The images are encoded as Numpy arrays, the labels are an array of digits from 0 to 9
# Training data (num,rows,cols):
print(train_images.shape)
print(len(train_labels))
print(train_labels)

# Test data (num,rows,cols):
print(test_images.shape)
print(len(test_labels))
print(test_labels)

# =====================Build neural network architecture===================
from keras import models
from keras import layers

# The network consists on a secuence of two Dense layers
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
# This layer will return an array of 10 probability scores. Each score will be the probability that the current digit
# image belongs to one of the 10 digit classes.
network.add(layers.Dense(10, activation='softmax'))

# =================compilation for training =============================
# The compilation step will need:
# - A "loss" function to measure its performance on the training data
# - An "optimizer" through which the network will update itself based on the data it sees and the "loss" function
# - "Metrics" such as acurracy
network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# ==================Preparing the image data======================
# Transform the arrays of shape (60000, 28, 28) ot type 'unit8' with values in the [0, 255] interval into 'float32'
# arrays of shape (60000, 28, 28) with values between 0 and 1
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

# ==================Label encoding =============================
from keras.utils import to_categorical

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# ==========================model training ======================
network.fit(train_images, train_labels, epochs=5, batch_size=128)

# ==========================model testing ========================
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_loss:', test_loss)
print('test_acc:', test_acc)

# ===============prediction====================
import numpy as np

predictions = network.predict(test_images)
sample = predictions[1]
print(sample.shape)  # Each entry in predictions is a vector of length 46
print(np.sum(sample))  # All the coefficients in the vectors sum to 1
print(np.argmax(sample))  # The largest entry is the predicted class (class with the highest probability)

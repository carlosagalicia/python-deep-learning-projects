# Single-label, multiclass classification
# This program classifies Reuters newswires into 46 mutually exclusive topics

# ===========================================Load the dataset ===========================================
from keras.datasets import reuters

# Restrict data to the 10000 most frecuent words in data
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

print("training data: ", train_data.shape)
print("testing data:", test_data.shape)
print(train_data[10])

# ========== word index decoding to english words ==========
word_index = reuters.get_word_index()  # dictionary of words and its respective indexes
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])  # reverse the values for the keys
# string of decoded words joined by a space character ' ', starting by the 4th index because indexes 1, 2 and 3 are
# reserved for "padding", "start of sequence" and "unknown" which will be replaced for a '?' character
decoded_newswire = ' '.join([reverse_word_index.get(i-3, '?') for i in train_data[0]])
print(decoded_newswire)

# The label associated with an example is an integer between 0 and 45 a topic index:
print(train_labels[10])

# ==========================================vectorize train_data and test_data=================================
import numpy as np


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results


x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

print("x_train: ", x_train.shape)
print("x_test: ", x_test.shape)

from tensorflow.keras.utils import to_categorical

one_hot_train_labels = to_categorical(train_labels)
one_hot_test_labels = to_categorical(test_labels)

# ==================================== building the network =======================================================
# The difference between this problem and the binary classification is that the number of output classes has gone from
# 2 to 46. Because we are working with 46 outputs, using 16-dimensional dense layers in the model might be to limited to
# learn to separate 46 different classes, due to this, we will use 64-dimensional dense layers.

from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))

# Ending with a dense layer of 46 output elements in a vector for each input sample. Furthermore, a 'softmax' activation
# is used to ouput a probability distribution over the 46 different output classes
model.add(layers.Dense(46, activation='softmax'))

# ==================================================model compilation ===========================================
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
# If you are following the aproach of label encoding by casting them as an integer tensor the compie process changes:
"""model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['acc'])"""

# =====================================Approach validation ===========================================
# Setting apart 1000 samples in the training data to use as a validation set
x_val = x_train[:1000]
partial_x_train = x_train[1000:]

# Label handling
y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]
# Another alternative for label encoding would be to cast them as an integer tensor:
"""y_train = np.array(train_labels)
y_test = np.array(test_labels)
"""

# =======================================Model training ================================================
history = model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512, validation_data=(x_val, y_val))
history_dict = history.history
print(history_dict.keys())

# ======================================plotting =======================================
# Plotting of the training and validation losses
import matplotlib.pyplot as plt
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.clf()
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, 'bo', label='Training accuracy ')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Because there are too many epochs (20), the training data is being overoptimized, this means that on new data the
# model's loss will rise and its accuracy will decreace unexpectedly, to avoid this we decrease the number of training
# epochs to 9:
# ==================================== building the network =======================================================
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(partial_x_train, partial_y_train, epochs=9, batch_size=512, validation_data=(x_val, y_val))
results = model.evaluate(x_test, one_hot_test_labels)
print(results)

history_dict = history.history
print(history_dict.keys())

# ======================================plotting =======================================
# Plotting of the training and validation losses
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.clf()
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, 'bo', label='Training accuracy ')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Generate the prediction for a sample
predictions = model.predict(x_test)
sample = predictions[0]
print(sample)
print(sample.shape)  # Each entry in predictions is a vector of length 46
print(np.sum(sample))  # All the coefficients in the vectors sum to 1
print(np.argmax(sample))  # The largest entry is the predicted class (class with the highest probability)


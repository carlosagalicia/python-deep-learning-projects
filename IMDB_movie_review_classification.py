# binary classification
# This program classifies movie negative and positive reviews from the Internet Movie Database (IMDB)

# =============================================Load the dataset ====================================
from keras.datasets import imdb
import numpy as np

# Only keeping the top 10000 most frequent words in the training data
# Train_data and test_data are vectors containing word indices expressed in numbers
# Train_labels and test_labels are vectors containing values of 1s and 0s, where 0 is negative and 1 is positive
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
print(train_data.shape)
print(train_labels.shape)
print(test_data.shape)
print(test_labels.shape)

# Because we only look the top 10000 most frecuent words, no word index will exceed 10000, reaching a max of 9999
print(max([max(sequence) for sequence in train_data]))

# ========== word index decoding to english words ==========
"""word_index = imdb.get_word_index()"""  # dictionary of words and its respective indexes
"""reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])"""  # reverse the values for the keys
# string of decoded words joined by a space character ' ', starting by the 4th index because indexes 1, 2 and 3 are
# reserved for "padding", "start of sequence" and "unknown" which will be replaced for a '?' character
"""decoded_review = ' '.join([reverse_word_index.get(i-3, '?') for i in train_data[0]])
print(decoded_review)"""


# ===================================================Preparing the data ============================================
# Turning lists into tensors by turning the lists into vectors of 0s and 1s, of size 10000
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))  # create an all-zero matrix of shape (len(sequence), dimension)
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results


# ==========================================vectorize train_data and test_data=================================
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
print(x_train)
print("\n")
print(x_test)
print("\n")
# vectorize labels
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')
print(y_train)
print("\n")
print(y_test)

# ==================================== building the network =======================================================
# Architecture choice:
# - 2 intermediate layers with 16-dimensional representation each
# - A 3rd layer outputting the 1 scalar prediction regarding the sentiment of the current review
from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# ==================================================model compilation ===========================================
# Because this is a binary classification problem, and the output of the model is a probability, we use the "binary_
# crossentropy" loss
# We use the "rmsprop" optimizer
# We will monitor acurracy during training
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

# It is possible to configure the parameters of the optimizer:
"""
from keras import optimizers
model.compile(optimizer=optimizers.RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])
"""

# It is possible to pass a custom loss function or metric function:
"""
from keras import losses
from keras import metrics

model.compile(optimizer=optimizers.RMSprop(lr=0.001), loss=losses.binary_crossentropy, metrics=[metrics.binary_accuracy])
"""

# =====================================Approach validation ===========================================
# Creating a valitation set by setting apart 10000 samples from the original training data
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]
print(x_val.shape)
print(partial_x_train.shape)

# =======================================Model training ================================================
# Training the model for 20 epochs, in mini-batches of 512 samples, monitoring loss and accuracy on the 10000 samples
# set apart by passing the validation data as the validation_data argument
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512, validation_data=(x_val, y_val))

# looking at the "history" dictionary data about everything that happened during training
history_dict = history.history
print(history_dict.keys())
results = model.evaluate(partial_x_train, partial_y_train)  # showing loss and accuracy values
print(results)

# ======================================plotting =======================================
# Plotting of the training and validation losses
import matplotlib.pyplot as plt

history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs = range(1, len(loss_values) + 1)

plt.plot(epochs, loss_values, 'bo', label='Training loss')  # 'bo' means blue dot
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')  # 'b' means solid blue line
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plotting of the training and validation accuracy
plt.clf()
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

plt.plot(epochs, acc_values, 'bo', label='Training acc')
plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Because there are too many epochs (20), the training data is being overoptimized, this means that on new data the
# model's loss will rise and its accuracy will decrease unexpectedly, to avoid this we decrease the number of training
# epochs to 4:
# ==================================== building the network =======================================================
# Architecture choice:
# - 2 intermediate layers with 16-dimensional representation each
# - A 3rd layer outputting the 1 scalar prediction regarding the sentiment of the current review
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
# =======================================Model training ================================================
# Training the model for 4 epochs, in mini-batches of 512 samples, monitoring loss and accuracy on the 10000 samples
# set apart by passing the validation data as the validation_data argument
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(partial_x_train, partial_y_train, epochs=4, batch_size=512, validation_data=(x_val, y_val))
results = model.evaluate(x_test, y_test)  # showing loss and accuracy values
print(results)
history_dict = history.history
print(history_dict.keys())

# ======================================plotting =======================================
# Plotting of the training and validation losses
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs = range(1, len(loss_values) + 1)

plt.plot(epochs, loss_values, 'bo', label='Training loss modified')  # 'bo' means blue dot
plt.plot(epochs, val_loss_values, 'b', label='Validation loss modified')  # 'b' means solid blue line
plt.title('Training and validation loss modified')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plotting of the training and validation accuracy
plt.clf()
acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']

plt.plot(epochs, acc_values, 'bo', label='Training acc modified')
plt.plot(epochs, val_acc_values, 'b', label='Validation acc modified')
plt.title('Training and validation accuracy modified')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Generate the likelihood of reviews being positive
print(model.predict(x_test))

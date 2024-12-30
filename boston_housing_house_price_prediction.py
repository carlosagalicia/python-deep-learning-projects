# Regression model
# This program attempts to predict the median price of homes in a given Boston suburb in the mid-1970, given data points
# about the suburb at the time, such as the crime rate, the local property tax rate, etc.

from keras.datasets import boston_housing
import numpy as np

# ===========================================Load the dataset ===========================================
(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()
print(train_data.shape)
print(test_data.shape)
print(train_targets)

# ===================================================Preparing the data ============================================
# To deal with values that all take different ranges, for each column in the input data matrix we substract the mean of
# the column and we divide it by the standard deviation
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

test_data -= mean
test_data /= std

# ==================================== building the network =======================================================
from keras import models
from keras import layers


# Because we'll need to instantiate the same model multiple times, we use a function to build it
def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))  # Ending with 1 unit and no activation to avoid any range output contraints
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])  # 'mse' means 'mean absolute error'
    return model


# =====================================Approach validation ===========================================
# To validate a model while we keep adjusting its parameters, we use the 'K-fold' cross validation, in which we
# split the data into 'K' partitions (usually 4 or 5), instantiating K identical models, and training each one on K-1
# partitions while evaluating on the remaining partition. The validation score for the model used is the average of the
# K validation scores obtained
k = 4
num_val_samples = len(train_data) // k
num_epochs = 250
all_mae_histories = []
all_scores = []

for i in range(k):
    print('processing fold #', i)
    # Preparing validation data from partition #k
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]  # [0:4], [4:8], [8:12] and so on
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]  # [0:4], [4:8], [8:12] and so on

    # Preparing the training data from other partitions, concatenate[[0:4], [4:0]
    partial_train_data = np.concatenate([train_data[:i * num_val_samples], train_data[(i + 1) * num_val_samples:]],
                                        axis=0)
    partial_train_targets = np.concatenate([train_targets[:i * num_val_samples],
                                            train_targets[(i + 1) * num_val_samples:]], axis=0)

    # Keras model building
    model = build_model()

    # Training the model in silent mode, verbose = 0
    history = model.fit(partial_train_data, partial_train_targets, validation_data=(val_data, val_targets),
                        epochs=num_epochs, batch_size=1, verbose=0)
    history_dict = history.history
    print(history_dict.keys())
    mae_history = history.history['val_mae']
    all_mae_histories.append(mae_history)

    # Model evaluation on the training data
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)

print(all_scores)
print(np.mean(all_scores))  # Average loss
print("The average loss with 300 epochs is 2.3-2.8")

# Compute the average of the per-epoch MAE(Mean Absolute Error) scores for all folds
average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]

# ======================================plotting =======================================
# original plot
import matplotlib.pyplot as plt
plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()


# Plot with the first 10 data points omitted and replacing each point with an exponential moving average of the points
# in order to obtain a smooth curve
def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous*factor + point*(1-factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


smooth_mae_history = smooth_curve(average_mae_history[10:])

plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

# Because the model is overfitting after 80 epochs, we build another network
model = build_model()
model.fit(train_data, train_targets, epochs=50, batch_size=16, verbose=0)  # Trained on the entirety of the data
test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)
print(test_mae_score)

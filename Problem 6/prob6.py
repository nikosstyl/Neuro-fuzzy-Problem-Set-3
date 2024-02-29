import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


MAX_EPOCHS = 200

# Function to generate AR data
def generate_AR_data(n_samples, a, seed=None):
    np.random.seed(seed)  # for reproducibility, if desired
    U_t = np.random.uniform(0, 0.5, n_samples)
    X_t = np.zeros(n_samples)
    for t in range(6, n_samples):
        X_t[t] = U_t[t] + a[0]*U_t[t-1] + a[1]*U_t[t-2] + a[2]*U_t[t-3] + a[3]*X_t[t-4] + a[4]*X_t[t-5] + a[5]*X_t[t-6]
    return X_t


# Function to create the dataset with a look_back
def create_dataset(X, look_back=3):
    dataX, dataY = [], []
    for i in range(len(X) - look_back):
        a = X[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(X[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

# Parameters
# a1, a2, a3 = 0.5, -0.1, 0.2
# a = [0.5, -0.1, 0.2]
a = [5, 5, -0.5, -0.5, -0.5, -0.5]  # Include a_0 for U_t terms
look_back = 3
training_sample_sizes = [100, 200, 500, 1000, 2000]  # Different numbers of training samples to investigate
mse_results = []

# Define early stopping
early_stopping = EarlyStopping(monitor='loss', patience=10)

converged_epocs = []

for n_samples in training_sample_sizes:
    # Generate training samples
    X_t = generate_AR_data(n_samples + look_back, a)

    # Normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_t_scaled = scaler.fit_transform(X_t.reshape(-1, 1))

    # Create the dataset
    X_train, y_train = create_dataset(X_t_scaled, look_back)
    X_train = np.reshape(X_train, (X_train.shape[0], 1, look_back))

    # Define the GRU model
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Fit the model
    model.fit(X_train, y_train, epochs=MAX_EPOCHS, batch_size=1, callbacks=[early_stopping])

    if early_stopping.stopped_epoch is None:
        converged_epocs.append(MAX_EPOCHS)
    else:
        converged_epocs.append(early_stopping.stopped_epoch)

    # Generate new samples for evaluation
    new_X_t = generate_AR_data(300 + look_back, a)

    # Scale the new samples using the previously fitted scaler
    new_X_t_scaled = scaler.transform(new_X_t.reshape(-1, 1))

    # Create the new dataset for evaluation
    X_new, y_new = create_dataset(new_X_t_scaled, look_back)
    X_new = np.reshape(X_new, (X_new.shape[0], 1, look_back))

    # Predict the output using the trained model
    predicted_new = model.predict(X_new)

    # Calculate the mean squared error on the new samples
    mse_new_samples = np.mean((y_new - predicted_new.reshape(-1))**2)
    mse_results.append(mse_new_samples)

    # Output the MSE for the current number of samples
    print(f'Mean Squared Error for {n_samples} training samples: {mse_new_samples}', end='\n\n')

# Plot the results
plt.plot(training_sample_sizes, mse_results, marker='o')
plt.title('MSE vs. Number of Training Samples')
plt.xlabel('Number of Training Samples')
plt.ylabel('Mean Squared Error')
plt.grid(True)

plt.figure()
plt.plot(training_sample_sizes, converged_epocs, marker='o')
plt.xlabel('Number of Training Samples')
plt.ylabel('Number of Epochs')
plt.title('Number of Epochs vs. Number of Training Samples')
plt.grid(True)
plt.show()

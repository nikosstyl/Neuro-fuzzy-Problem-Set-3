import numpy as np
import matplotlib.pyplot as plt

# Define the target function
def target_function(p):
    return 1 + np.sin(np.pi * p / 8)

# Generate random data points within the interval [-4, 4]
np.random.seed(42)  # for reproducibility
data_points = np.random.uniform(-4, 4, 30)
targets = target_function(data_points)

def initialize_rbf_centers(num_centers, data_points):
    # Select random centers from the training data
    indices = np.random.choice(len(data_points), size=num_centers, replace=False)
    return data_points[indices]

def rbf_activation(data_point, center, spread=1.0):
    # Gaussian RBF
    return np.exp(-np.linalg.norm(data_point - center)**2 / (2 * spread**2))

def compute_rbf_outputs(data_points, centers, spread=1.0):
    # Compute the RBF activations for each data point for all centers
    rbf_outputs = np.array([[rbf_activation(p, c, spread) for c in centers] for p in data_points])
    return rbf_outputs

def forward_pass(rbf_outputs, weights, bias):
    # Calculate the weighted sum of RBF outputs and add bias
    network_output = np.dot(rbf_outputs, weights) + bias
    return network_output

def compute_loss(predictions, targets):
    # Calculate the sum of squared errors
    loss = np.sum((predictions - targets) ** 2)
    return loss

def train_rbf_network(data_points, targets, centers, learning_rate, epochs):
    # Initialize weights and bias to small random values
    weights = np.random.randn(len(centers))
    bias = np.random.randn(1)

    # Training loop
    for epoch in range(epochs):
        # Compute RBF outputs
        rbf_outputs = compute_rbf_outputs(data_points, centers)

        # Forward pass
        predictions = forward_pass(rbf_outputs, weights, bias)

        # Compute loss
        loss = compute_loss(predictions, targets)
        print(f'Epoch {epoch+1}, Loss: {loss:.4f}', end='\r')
        if loss > 1e4:
            print('Loss exploded, stopping training', end='\n\n')
            raise ValueError(f'Loss exploded: {loss:.4f} > 1e4')
        elif loss < 3e-2:
            print('Converged, stopping training', end='\n\n')
            break

        # Compute gradients
        error = predictions - targets
        weights_gradient = np.dot(rbf_outputs.T, error)
        bias_gradient = np.sum(error)

        # Update weights and bias - Steepest Descent
        weights -= learning_rate * weights_gradient
        bias -= learning_rate * bias_gradient
    print('')

    return weights, bias

def compute_rbf (learning_rate, num_centers, epochs=2000):
    # Train the RBF network
    centers = initialize_rbf_centers(num_centers, data_points)
    weights, bias = train_rbf_network(data_points, targets, centers, learning_rate, epochs)

    # Plot the trained network response
    rbf_outputs = compute_rbf_outputs(p_values, centers)
    network_response = forward_pass(rbf_outputs, weights, bias)

    # Compute the sum squared error over the training set
    train_predictions = forward_pass(compute_rbf_outputs(data_points, centers), weights, bias)
    final_loss = compute_loss(train_predictions, targets)
    print(f'Final sum squared error over the training set: {final_loss:.4f}')

    return network_response


# # Parameters
lr = [0.001, 0.01, 0.1]
num_centers = [4, 8, 12, 20]

fig, (target_plt, train_plt) = plt.subplots(2,1, figsize=(10, 7))
p_values = np.linspace(-4, 4, 100)

# Plot the function and the selected data points
target_plt.plot(p_values, target_function(p_values), label='Target function')
target_plt.scatter(data_points, targets, color='red', label='Training points')
target_plt.legend()
target_plt.grid(True)
target_plt.set_xlabel('p')
target_plt.set_ylabel('g(p)')
target_plt.set_title('Target function and training data points')

response = compute_rbf(0.01, num_centers=12, epochs=int(2e4))

# Plot the network response
train_plt.plot(p_values, target_function(p_values), label='Target function')
train_plt.plot(p_values, response, label='Network response', linestyle='dashed')
train_plt.scatter(data_points, targets, color='red', label='Training points')
train_plt.legend()
train_plt.set_xlabel('p')
train_plt.set_ylabel('g(p)')
train_plt.set_title('Target function and RBF network response')

fig.tight_layout()
plt.show()
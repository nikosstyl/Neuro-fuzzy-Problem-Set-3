import numpy as np
import matplotlib.pyplot as plt

# Define the input vectors
p1 = np.array([1, 1])
p2 = np.array([-1, 2])
p3 = np.array([-2, -2])

# Define the learning rate
alpha = 0.5

threshold = 1e-5

# Initialize the weight vectors for the two neurons
w2 = np.array([1, 0])
w1 = np.array([0, 1])

# Define the order of presentation of the vectors
order = [p1, p2, p3, p2, p3, p1]

# Train the competitive layer
for epoch in range(100):  # Number of epochs
    w1_old = w1.copy()
    w2_old = w2.copy()
    for p in order:
        # Calculate the distances
        d1 = np.linalg.norm(p - w1)
        d2 = np.linalg.norm(p - w2)

        # Determine the winning neuron
        if d1 < d2:
            # Update the weight vector of the winning neuron
            w1 = w1 + alpha * (p - w1)
        else:
            w2 = w2 + alpha * (p - w2)
        
    plt.plot(epoch, w1[0], 'o', color='r')
    plt.plot(epoch, w1[1], 'x', color='b')
    plt.plot(epoch, w2[0], '*', color='g')
    plt.plot(epoch, w2[1], '.', color='orange')
    
    if np.linalg.norm(w1 - w1_old) < threshold and np.linalg.norm(w2 - w2_old) < threshold:
        print(f'Converged at epoch {epoch}')
        break

print('W1:')
print(w1, end='\n\n')
print('W2:')
print(w2, end='\n\n')

plt.xlabel('Epoch')
plt.ylabel('Weights')
plt.title('Weight vectors over epoch')
plt.grid()
plt.legend(['w1[0]', 'w1[1]', 'w2[0]', 'w2[1]'])
plt.show()
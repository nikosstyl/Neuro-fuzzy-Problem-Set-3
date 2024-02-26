import numpy as np
import matplotlib.pyplot as plt

USE_CONV_CHECK = True

EPOCHS = 1000

# Define the input vectors
p1 = np.array([2, 0])
p2 = np.array([0, 1])
p3 = np.array([2, 2])

# Define the learning rate
alpha = 0.5

threshold = 1e-5

# Initialize the weight vectors for the two neurons
w1 = np.array([1, 0])
w2 = np.array([-1, 0])

# Define the order of presentation of the vectors
order = [p1, p2, p3, p2, p3, p1]
    
def plot_weights(epoch, w1, w2):
    plt.plot(epoch, w1[0], 'o', color='r')
    plt.plot(epoch, w1[1], 'x', color='b')
    plt.plot(epoch, w2[0], '*', color='g')
    plt.plot(epoch, w2[1], '.', color='orange')

w1_changes = []
w2_changes = []

# Train the competitive layer
for epoch in range(EPOCHS):  # Number of epochs
    print(f'Epoch: {epoch}', end='\r')
    w1_old = w1.copy()
    w2_old = w2.copy()
    # plot_weights(epoch, w1, w2)

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
        
    w1_changes.append(w1.copy())
    w2_changes.append(w2.copy())
    
    if np.linalg.norm(w1 - w1_old) < threshold and np.linalg.norm(w2 - w2_old) < threshold and USE_CONV_CHECK == True:
        # plot_weights(epoch, w1, w2)
        print(f'\nConverged at epoch {epoch}')
        break

print('\n\nW1:')
print(w1, end='\n\n')
print('W2:')
print(w2, end='\n\n')

w1_changes = np.array(w1_changes)
w2_changes = np.array(w2_changes)

plt.plot(p1[0], p1[1], 'o')
plt.plot(p2[0], p2[1], 'rx')
plt.plot(p3[0], p3[1], 'o')
plt.plot(w1_changes[:, 0], w1_changes[:, 1], 'o-')
plt.plot(w2_changes[:, 0], w2_changes[:, 1], 'o-')
plt.grid()
plt.legend(['Initial $p_1$', 'Initial $p_2$', 'Initial $p_3$', '$w_1$', '$w_2$'])
plt.xlabel('Dim 1')
plt.ylabel('Dim 2')
plt.title('Trajectories of weights')
plt.show()
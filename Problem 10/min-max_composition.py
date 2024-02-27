import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the parameters
k = 1  # Example value
x = np.linspace(-10, 10, 100)
z = np.linspace(-10, 10, 100)
X, Z = np.meshgrid(x, z)

# Define the membership functions
def mu_R1(x, y):
    return np.exp(-k * (x - y)**2)

def mu_R2(y, z):
    return np.exp(-k * (y - z)**2)

# Calculate the max-min composition
Y = np.linspace(-10, 10, 100)
mu_R = np.zeros_like(X)
for i in range(len(Y)):
    mu_R = np.maximum(mu_R, np.minimum(mu_R1(X, Y[i]), mu_R2(Y[i], Z)))

# Plot the max-min composition
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Z, mu_R, cmap='viridis')  # Assign the result to surf
ax.set_xlabel('X axis')
ax.set_ylabel('Z axis')
ax.set_zlabel('μR(x,z)')
ax.set_title('Max-Min Composition of R1 and R2')

# Show the plot with a colorbar
fig.colorbar(surf)  # Use fig.colorbar instead of plt.colorbar
plt.show()

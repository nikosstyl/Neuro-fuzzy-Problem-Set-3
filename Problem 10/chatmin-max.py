import numpy as np
import matplotlib.pyplot as plt

# Define the membership functions
def mu_R1(x, y, k=1):
    return np.exp(-k * (x - y) ** 2)

def mu_R2(y, z, k=1):
    return np.exp(-k * (y - z) ** 2)

# Define the range for x, y, and z
x_values = np.linspace(-10, 10, 100)
z_values = np.linspace(-10, 10, 100)
y_values = np.linspace(-10, 10, 100)

# Initialize the matrix to hold the max-min composition values
mu_R = np.zeros((len(x_values), len(z_values)))

# Calculate the max-min composition
for i, x in enumerate(x_values):
    for j, z in enumerate(z_values):
        min_values = np.minimum(mu_R1(x, y_values), mu_R2(y_values, z))
        mu_R[i, j] = np.max(min_values)

# Plot the max-min composition
X, Z = np.meshgrid(x_values, z_values)
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
surf = ax.plot_surface(X, Z, mu_R.T, cmap='viridis', edgecolor='none')

# Labels and title
ax.set_xlabel('X axis')
ax.set_ylabel('Z axis')
ax.set_zlabel('Membership Degree')
ax.set_title('Max-Min Composition of Fuzzy Relations R1 and R2')

# Show the plot
plt.colorbar(surf)
plt.show()

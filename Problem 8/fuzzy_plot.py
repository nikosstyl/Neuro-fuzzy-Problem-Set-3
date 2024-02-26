import numpy as np
import matplotlib.pyplot as plt

# Calculate the radius squared for the level curve
level = 0.3
radius_squared = 1/0.7 - 1

# Create a grid of points
x = np.linspace(-3, 3, 400)
y = np.linspace(-3, 3, 400)
X, Y = np.meshgrid(x, y)

# Calculate the membership function values
Z = 1 - 1 / (1 + X**2 + Y**2)

# Plot the contour for the level 0.3
plt.figure(figsize=(8, 8))
plt.contour(X, Y, Z, levels=[level], colors='blue')
plt.fill_between(x, np.sqrt(radius_squared), 3, color='skyblue', alpha=0.5)
plt.fill_between(x, -np.sqrt(radius_squared), -3, color='skyblue', alpha=0.5)

# Formatting the plot
plt.title('Ordinary Relation of Level 0.3 for the Fuzzy Relation')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.axis('equal')

# Show the plot
plt.show()

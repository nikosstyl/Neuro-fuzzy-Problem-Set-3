import numpy as np
import matplotlib.pyplot as plt

# Define alpha and calculate corresponding threshold
alpha = 0.3
threshold = 1 / (1 - alpha)

# Define range for x and y
x = np.linspace(-3, 3, 400)
y = np.linspace(-3, 3, 400)

# Create meshgrid for x and y
X, Y = np.meshgrid(x, y)

# Calculate membership function for each point in the meshgrid
mu_R = 1 - 1 / (1 + X**2 + Y**2)

# Plot the contour of the membership function
plt.contourf(X, Y, mu_R, levels=[alpha, 1], colors=['skyblue', 'white'])

# Plot the circle x^2 + y^2 = 3/7
theta = np.linspace(0, 2*np.pi, 100)
plt.plot(np.sqrt(3/7) * np.cos(theta), np.sqrt(3/7) * np.sin(theta), 'r--', label='Circle: $x^2 + y^2 = \\frac{3}{7}$')

# Labeling and customization
plt.title('Ordinary Relation of Level 0.3 for Fuzzy Relation')
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')
plt.legend()
plt.grid(True)

# Show plot
plt.show()

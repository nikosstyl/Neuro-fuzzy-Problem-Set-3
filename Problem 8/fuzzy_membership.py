import numpy as np
import matplotlib.pyplot as plt

# Calculate the radius
r = np.sqrt(1 / 0.7 - 1)

# Generate values for theta
theta = np.linspace(0, 2*np.pi, 100)

# Generate coordinates for the points on the circle
x = r * np.cos(theta)
y = r * np.sin(theta)

# Plotting
fig, ax = plt.subplots(figsize=(6,6))
ax.fill(x, y, 'b', alpha=0.3, label='Level ≥ 0.3')
ax.plot(x, y, 'b-', linewidth=2)  # Circle boundary
ax.grid(True, which='both')
ax.set_aspect('equal', 'box')
ax.axhline(y=0, color='k')
ax.axvline(x=0, color='k')

# Labels and Title
plt.title('Ordinary Relation of Level 0.3')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

# Calculate the area
area = np.pi * r**2
print("The area enclosed by the circle is:", area)

# Show the plot
plt.show()
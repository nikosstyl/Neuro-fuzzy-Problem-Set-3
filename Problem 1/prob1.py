import matplotlib.pyplot as plt

# Define the centers and approximate radii of the circles
centers = [(-1, 1.5), (2, 2)]
radii = [0.5, 0.25]  # Assumed radii, adjust as needed

# Create a figure and a set of subplots
fig, ax = plt.subplots()

# Add circles to the plot
for center, radius in zip(centers, radii):
    circle = plt.Circle(center, radius, color='blue', fill=True, alpha=0.5)
    ax.add_artist(circle)

# Set aspect of the plot to be equal
ax.set_aspect('equal')

# Set limits of the plot
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(0, 2.5)

# Label the axes
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')

# Show grid
ax.grid(True)

# Show the plot with circles
plt.show()

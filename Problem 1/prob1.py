import matplotlib.pyplot as plt

# Define the centers and approximate radii of the circles
centers = [(-1, 1.5), (2, 2)]
radii = [0.5, 0.25]  # Assumed radii, adjust as needed


# Add circles to the plot
for center, radius in zip(centers, radii):
    circle = plt.Circle(center, radius, fill=True, alpha=0.5)
    plt.gca().add_patch(circle) 

plt.gca().set_aspect('equal')

plt.xlim(-2.5, 2.5)
plt.ylim(0, 4)

# Label the axes
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# Show grid
plt.grid(True)

plt.tight_layout()

# Show the plot with circles
plt.show()

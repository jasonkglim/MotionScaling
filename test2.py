import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns

# Step 1: Prepare your data
data = np.random.rand(10, 10)  # Random data for demonstration

# Step 2: Set up the plot
fig, ax = plt.subplots()
sns.heatmap(data, ax=ax)

# Step 3: Create a function to update the heatmap
def update(*args):
    data = np.random.rand(10, 10)  # Generate new random data
    ax.clear()  # Clear the previous heatmap
    sns.heatmap(data, ax=ax, cbar=False)  # Create a new heatmap

# Step 4: Use FuncAnimation to create the animation
ani = animation.FuncAnimation(fig, update, frames=30, interval=200)

# Step 5: Save or show the animation
plt.show()  # Show the animation
# ani.save('heatmap_animation.mp4')  # Uncomment to save the animation as a file

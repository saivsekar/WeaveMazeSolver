import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd

# Load the CSV maze file
maze = pd.read_csv('solved_maze.csv', header=None).to_numpy()

# Set up the figure and axis
fig, ax = plt.subplots()
ax.set_xticks([])
ax.set_yticks([])

# Initialize the maze display with numerical color codes
color_map = {
    1: [0, 0, 0],    # Black for walls
    0: [1, 1, 1],    # White for open spaces
    5: [1, 1, 1]     # Initially white for solution path
}
maze_colors = np.zeros((*maze.shape, 3))  # Create an RGB array
for value, color in color_map.items():
    maze_colors[maze == value] = color

image = ax.imshow(maze_colors, interpolation='nearest')

# Find the path of the solution
path = np.argwhere(maze == 5)

# Animation update function
def update(frame):
    # Update the current cell in the path to green
    row, col = path[frame]
    maze_colors[row, col] = [0, 1, 0]  # Green for solution path
    image.set_data(maze_colors)  # Update the image data

# Create the animation
ani = FuncAnimation(fig, update, frames=len(path), interval=100, repeat=False)

plt.show()
ani.save('mazevid.gif', writer='pillow', fps = 20 ) 
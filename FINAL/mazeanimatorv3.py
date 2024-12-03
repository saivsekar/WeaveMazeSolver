"""
Created on Nov 20 22:24:29 2024

@author: saivchandrasekar
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import ast

# load in solution path
with open('solpath.txt', 'r') as file:
    solution_path = [ast.literal_eval(line.strip()) for line in file]

# Load the CSV maze file
maze = np.loadtxt('solved_maze.csv', delimiter=',')

# Set up the figure and axis
fig, ax = plt.subplots()
ax.set_xticks([])
ax.set_yticks([])

# Initialize the maze display with numerical color codes
color_map = {
    1: [0, 0, 0],    # Black for walls
    0: [1, 1, 1],    # White for open spaces
    5: [1, 1, 1]     # Initially white for solution path (before frame)
}

# Create rgb array (mapped onto csv)
maze_colors = np.zeros((*maze.shape, 3)) 
for value, color in color_map.items():
    maze_colors[maze == value] = color

image = ax.imshow(maze_colors, interpolation='nearest')

# Animation update function (from assignment 3)
def update(frame):

    # Update the current cell in the solution path to green
    row, col = solution_path[frame]
    # solution path is turned green pixel by pixel
    maze_colors[row, col] = [0, 1, 0]
    # update image data (1 pixel) for this frame
    image.set_data(maze_colors) 

# Create the animation
ani = FuncAnimation(fig, update, frames=len(solution_path), interval=100, repeat=False)

# Save the animation as a GIF
ani.save('maze_solution.gif', writer='pillow', fps=10)

plt.show()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 22:24:29 2024

@author: vaibhavgoggela
"""

import cv2
import numpy as np
import csv

# Load the maze image
image_path = 'maze.png'  # Update this path if necessary
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Convert to binary image (0 for black, 1 for white)
_, binary_image = cv2.threshold(image, 128, 1, cv2.THRESH_BINARY_INV)

# Define the new cell size
cell_size = 3
grid_size = (image.shape[0] // cell_size, image.shape[1] // cell_size)  # (26, 26)

# Resize the image to a 26x26 matrix
resized_maze_matrix = cv2.resize(binary_image, grid_size, interpolation=cv2.INTER_AREA)

# Define the pattern to search for and the replacement pattern
pattern = [1, 0, 1, 1, 1, 0, 1]
replacement_pattern = [2, 0, 1, 1, 1, 0, 2]

# Search for the pattern in each row
for i in range(resized_maze_matrix.shape[0]):
    row = resized_maze_matrix[i, :]
    # Find occurrences of the pattern in the row
    for j in range(len(row) - len(pattern) + 1):
        if np.array_equal(row[j:j+len(pattern)], pattern):
            # Replace the pattern with the modified pattern
            resized_maze_matrix[i, j:j+len(pattern)] = replacement_pattern

# Search for the pattern in each column
for j in range(resized_maze_matrix.shape[1]):
    column = resized_maze_matrix[:, j]
    # Find occurrences of the pattern in the column
    for i in range(len(column) - len(pattern) + 1):
        if np.array_equal(column[i:i+len(pattern)], pattern):
            # Replace the pattern with the modified pattern
            resized_maze_matrix[i:i+len(pattern), j] = replacement_pattern

# Display the maze matrix in the console
print("Maze Matrix (1 for paths, 0 for walls):")
print(resized_maze_matrix)

# Save the modified maze matrix as a CSV file
csv_file_path = 'maze_matrix.csv'
with open(csv_file_path, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerows(resized_maze_matrix)
print(f"Maze matrix saved as CSV file at '{csv_file_path}'.")

# Optional: save the modified matrix as an image for visual verification
output_image = (resized_maze_matrix * 85).astype(np.uint8)  # Scale 0-255 for 0, 1, 2
cv2.imwrite('maze_matrix_output.jpg', output_image)
print("Maze matrix also saved as 'maze_matrix_output.jpg'.")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 21:14:44 2024

@author: haojijiang
"""

import numpy as np
import pandas as pd
import heapq
import matplotlib.pyplot as plt
solpath = 0

def h_cost(a, b): # Calculates the Manhattan Distance
    x_distance = abs(a[0] - b[0])
    y_distance = abs(a[1] - b[1])
    distance = x_distance + y_distance
    return distance 

def a_search(start, goal, grid):

    queue = []
    heapq.heappush(queue, (0, start))  # Priority queue

    predecessor = {}
    g = {start: 0} # G function
    f = {start: h_cost(start, goal)} # F function
    
    
    underpass = {}
    
    # Find all '2's in the grid and tuple their coordinates 
    positions = np.argwhere(grid == 2)

    positions = [tuple(pos) for pos in positions]
    
    # Find underpass pairs and add them to a dictionary
    
    for k in positions:
        #Right Hand Side
        if (grid[k[0] - 1][k[1]] == 2 or grid[k[0] + 1][k[1]] == 2) and grid[k[0]][k[1] + 1] == 1:
            underpass[k] = (k[0], k[1] + 6)
        
        #Left Hand Side
        if (grid[k[0] - 1][k[1]] == 2 or grid[k[0] + 1][k[1]] == 2) and grid[k[0]][k[1] - 1] == 1:
            underpass[k] = (k[0], k[1] - 6)
                            
        #Top to bottom
        if (grid[k[0]][k[1] - 1] == 2 or grid[k[0]][k[1] + 1] == 2) and grid[k[0] + 1][k[1]] == 1:
            underpass[k] = (k[0] + 6, k[1])
        
        #Bottom to Top
        if (grid[k[0]][k[1] - 1] == 2 or grid[k[0]][k[1] + 1] == 2) and grid[k[0] - 1][k[1]] == 1:
            underpass[k] = (k[0] - 6, k[1])

    
    

    while queue:
        current = heapq.heappop(queue)[1]

        if current == goal:
            # Reconstruct path
            path = []
            
            while current in predecessor:
                path.append(current)
                current = predecessor[current]
            path.append(start)
            return path[::-1]  # Return reversed path
        
        
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)] # Neighbors 

        
        if grid[current[0]][current[1]] == 0:
        
        
            for y, x in directions:  
                neighbor = (current[0] + y, current[1] + x)
                if (0 <= neighbor[0] < grid.shape[0] and 0 <= neighbor[1] < grid.shape[1]):
                    if grid[neighbor[0]][neighbor[1]] in (0,2):
                
                        g_score = g[current] + 1  # Cost to move to neighbor
    
                        if neighbor in g:
                            temp_g_score = g[neighbor]
                        else:
                            temp_g_score = float('inf')
                        
                        if g_score < temp_g_score:
                            predecessor[neighbor] = current
                            g[neighbor] = g_score
                            h_score = h_cost(neighbor, goal)
                            f[neighbor] = g_score + h_score
        
                            if neighbor not in [i[1] for i in queue]:
                                heapq.heappush(queue, (f[neighbor], neighbor))


        elif grid[current[0]][current[1]] == 2:
            
            for y, x in directions.append((underpass[current][0] - current[0],underpass[current][1] - current[1])):
                neighbor = (current[0] + y, current[1] + x)
                
                if (0 <= neighbor[0] < grid.shape[0] and 0 <= neighbor[1] < grid.shape[1]):
                    
                    if grid[neighbor[0]][neighbor[1]] in (0,2):
                
                        g_score = g[current] + 6  # Cost to move to neighbor
    
                        if neighbor in g:
                            temp_g_score = g[neighbor]
                        else:
                            temp_g_score = float('inf')
                        
                        if g_score < temp_g_score:
                            predecessor[neighbor] = current
                            g[neighbor] = g_score
                            h_score = queue(neighbor, goal)
                            f[neighbor] = g_score + h_score
        
                            if neighbor not in [i[1] for i in queue]:
                                heapq.heappush(queue, (f[neighbor], neighbor))
                            
            
    return [] 

def convert_maze(file_path):
    maze = pd.read_csv(file_path, header=None)
    return maze.to_numpy()

def new_csv(maze, file_path):
    maze = pd.DataFrame(maze)
    maze.to_csv(file_path, index=False, header=False)

def visualize(maze, path): # Delete if not needed
    maze = maze.copy()
    for (x, y) in path:
        maze[x, y] = 5  # Mark the path with a 2

    plt.imshow(maze, cmap='hot', interpolation='nearest')
    plt.title("Maze with Solution Path")
    plt.axis('off')
    plt.show()
    plt.savefig(maze)


if __name__ == "__main__":
    
    maze_file = 'maze_matrix.csv'
    maze = convert_maze(maze_file)

    # Define start and end points
    
    starter_maze = maze.copy()
    
    rows = np.shape(starter_maze)[0] - 1
    cols = np.shape(starter_maze)[1] - 1
    
    flag = True

    
    # Search first column
    
    for i in range(0, rows):
        if starter_maze[i, 0] == 0 and starter_maze[i + 1, 0] == 0 and starter_maze[i + 2, 0] == 0:
            start = (i + 1, 0)
            starter_maze[i, 0] = 1
            starter_maze[i + 1, 0] = 1
            starter_maze[i + 2, 0] = 1
            flag = False
            break
        
    # Search last column
    
        elif starter_maze[i, cols] == 0 and starter_maze[i + 1, cols] == 0 and starter_maze[i + 2, cols] == 0:
            start = (i, cols)
            starter_maze[i, cols] = 1
            starter_maze[i + 1, cols] = 1
            starter_maze[i + 2, cols] = 1
            flag = False
            break
        
    # Search first row
    
    if flag:
        for i in range(0, cols):
            if starter_maze[0, i] == 0 and starter_maze[0, i + 1] == 0 and starter_maze[0, i + 2] == 0:
                start = (0, i)
                starter_maze[0, i] = 1
                starter_maze[0, i + 1] = 1
                starter_maze[0, i + 2] = 1
                break
            
    # Search last row
    
            elif starter_maze[rows, i] == 0 and starter_maze[rows, i + 1] == 0 and starter_maze[rows, i + 2] == 0:
                start = (rows, i)
                starter_maze[rows, i] = 1
                starter_maze[rows, i + 1] = 1
                starter_maze[rows, i + 2] = 1
                break
            
    # Find exit 
    
    for i in range(0,rows):
        if starter_maze[i, 0] == 0:
            goal = (i, 0)
            break
        elif starter_maze[i, cols] == 0:
            goal = (i, cols)
            break
        
    for i in range(0, cols):
        if starter_maze[0, i] == 0:
            goal = (0, i)
            break
        elif starter_maze[rows, i] == 0:
            goal = (rows, i)
            break
  

    # Perform A* search
    path = a_search(start, goal, maze)
    solpath = path
    if path:
        print("Path found:", path)
        
        # Mark the path in the maze
        for (x, y) in path:
            maze[x, y] = 5  # Can be modified

        # Write the modified maze to a new CSV file
        output_file = 'solved_maze.csv'  
        new_csv(maze, output_file)
        print(f"Solution saved to {output_file}")
        
        # Write solution path to text file
        with open('solpath.txt', 'w') as file:
            file.write("\n".join(str(tup) for tup in path))


    else:
        print("No path found.")

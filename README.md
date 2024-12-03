# MazeSolver
ME396P Final Project

This is team Rocket's final project repository. Here, we endeavor to solve the problem of solving a basic 'Weave Maze'. A 'Weave Maze' is a maze where the corridors are able to go over/under one another. It can be said that a weave maze could be considered as three-dimensional.

## Open CV
This .py file converts a 2D maze image into a grid of 1's and 0's, then writes it to a csv file along with a basic visualization. 

## Pathfinder
This file identifies 'weave' locations and implements the A* pathfinding algorithm to find the best solution path for the given maze. 

## Mazeanimatorv2
This file implements matplotlib animation functions to create an animation of the solution path overlapping the maze image. 

## Procedure
Go to the FINAL folder
First, feed in a 2d image of a maze named 'maze.png' to the OpenCV.py program. Then run pathfinder.py to generate a solution path. Finally, run mazeanimatorv2.py to create a .gif animation of the maze solution.

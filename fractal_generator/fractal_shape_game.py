import numpy as np
import matplotlib.pyplot as plt
import random


#plot a closed shape with the given verticies
def plot_shape(verticies):
    ''' plots a closed shape with the given verticies '''

    x = []
    y = []

    plt.style.use("dark_background")
    for i in range(len(verticies)):
        x.append(verticies[i][0])
        y.append(verticies[i][1])
    x.append(verticies[0][0])
    y.append(verticies[0][1])
    plt.plot(x,y, 'white')

#plot a point on the shape
def plot_point(point):
    ''' plots a single point '''

    plt.plot(point[0],point[1], 'ro', markersize=0.25)

def complex_shape_game(shape, start, turns, distance=2/3):

    ''' plays the game, which consists of starting at a point (start),
    picking a random vertex of the shape (shape), 
    and moving towards that vertex (distance), 
    (turns) number of times, plotting the points along the way '''

    #plot the shape
    plot_shape(shape)

    #plot the starting point
    plot_point(start)

    #set the current point to the starting point
    current_point = start 
   
    #create empty array to store points
    points = []

    for i in range(turns):
        #pick a random vertex
        vertex = shape[random.randint(0, len(shape)-1)]

        
        #find the point 'distance' of the way to the vertex
        current_point = [current_point[0] + distance*(vertex[0]-current_point[0]),
        current_point[1] + distance*(vertex[1]-current_point[1])]

        #add point to points
        points.append([current_point, vertex])
    
    # plot all of the points
    plt.scatter([point[0][0] for point in points], [point[0][1] for point in points], c='red', s=0.25, alpha=1)

    #show the plot
    plt.show()

    #close the plot
    plt.close()

    return points

def fractal_zoom(num_frames, shape, points):

    ''' after running complex_shape_game and generating a list of points (points), 
    this function will zoom in on the shape for (num_frames) frames, highlighting the fractal nature of the shape. 
    Currently only zooms in on the point (0,0) '''

    xlim = [0, 1]
    ylim = [0, 1]

    x_increment = (xlim[1] - xlim[0]) / num_frames
    y_increment = (ylim[1] - ylim[0]) / num_frames

    #plt.scatter([point[0] for point in points], [point[1] for point in points], c='red', s=0.25)
    #plt.show()
    for i in range(num_frames):
        plot_shape(shape)

        plt.xlim([xlim[0], xlim[1] - x_increment * i])
        plt.ylim([ylim[0], ylim[1] - y_increment * i])

        plt.scatter([point[0][0] for point in points], [point[0][1] for point in points], c='red', s=1, alpha=1)
        
        plt.savefig(f"/Users/nicklatina/Desktop/complex_shape_zoom/{1000 + i}_complex_shape_zoom.png")
        plt.clf()






pentagon = [[0.2,0],[0,0.65],[0.5,1], [1, 0.65], [0.8, 0]]

triangle = [[0,0],[0.5,1],[1,0]]

house = [[0,0],[0,0.6],[0.5,1], [1, 0.6], [1, 0]]

starting_point = [0.5, 0.2]



complex_shape_game(house, starting_point, turns=120000, distance=2/3)

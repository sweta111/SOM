# Imports
import numpy as np
import math
from PIL import Image
from pylab import *
import matplotlib.pyplot as plt
from numpy import linalg as LA
class SOM:
    def __init__(self, x_size, y_size, dimen, num_iter, t_step):
        # init weights to 0 < w < 256
        self.weights = np.random.randint(256, size=(x_size, y_size, dimen))\
                            .astype('float64')
        self.num_iter = num_iter
        self.map_radius = max(self.weights.shape)/2 # sigma_0
        self.t_const = self.num_iter/math.log(self.map_radius) # lambda
        self.t_step = t_step
        
    def get_bmu(self, vector):
        # calculate euclidean dist btw weight matrix and vector
        distance = np.sqrt(np.sum((self.weights - vector) ** 2, 2))
#        a = np.exp(-distance / np.max(distance))
#        distance = np.exp(-distance/50000)
#        imshow(distance)
#        distance = a
        min_idx = distance.argmin()
        return np.unravel_index(min_idx, distance.shape), distance
        
    def get_bmu_dist(self, vector):
        # initialize array where values are its index
        x, y, rgb = self.weights.shape
        xi = np.arange(x).reshape(x, 1).repeat(y, 1)
        yi = np.arange(y).reshape(1, y).repeat(x, 0)
        # returns matrix of distance of each index in 2D from BMU
        bmu, distance = self.get_bmu(vector)
        return distance, np.sum((np.dstack((xi, yi)) - np.array(bmu)) ** 2, 2)

    def get_nbhood_radius(self, iter_count):
        return self.map_radius * np.exp(-iter_count/self.t_const)
        
    def teach_row(self, vector, i):
        nbhood_radius = self.get_nbhood_radius(i)
        distance, bmu_dist = self.get_bmu_dist(vector)
        bmu_dist = bmu_dist.astype('float64')
        # exponential decaying learning rate
        lr = 0.1 * np.exp(-i/self.num_iter) 
        
        # influence
        theta = lr * np.exp(-(bmu_dist)/ (2 * nbhood_radius ** 2))
        return distance, np.expand_dims(theta, 2) * (vector - self.weights)
        
    def teach(self, t_set, map_w, map_h, img_x, img_y):
        distances = []
        error = []
        total_error = []
        for i in range(self.num_iter):
            error = []
            for j in range(len(t_set)):
                distance, teach_row = self.teach_row(t_set[j], i)
                distances.append(distance)
                self.weights += teach_row
                error.append(LA.norm((t_set[j] - self.weights), axis = 2)) 
            total_error.append(np.mean(error)) 
            
            if i % 10 == 0:
                print("Training Iteration: ", i, "total_error", total_error[i])
                
            if(i == 0):
                weight_saved = self.weights
                distance_saved = distance
                weight_iter = i
            else:
                if(total_error[i] < total_error[weight_iter]):
                    weight_saved = self.weights
                    distance_saved = distance
                    weight_iter = i
            
        return distances, error, total_error, weight_saved, weight_iter, distance_saved
        
    def show(self):
        im = Image.fromarray(self.weights.astype('uint8'), mode='RGB')
        im.format = 'JPG'
        im.show()
        

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# Using first 10000 images for training
train_data = mnist.train.images[:100,:]
img_x = 28
img_y = 28
data_length = 100       

# Converting normalized data to values between 0 and 256
train_data = train_data * 256

# Hyperparameters
map_w = 21
map_h = 21
data_dimens = img_x * img_y
epochs = 100
t_step = 1

# Defining Map
mnist_map = SOM(map_w, map_h, data_dimens, epochs, t_step)

# Start Training
distances, error, total_error, weight_saved, weight_iter, distance_saved = mnist_map.teach(train_data, map_w, map_h, img_x, img_y)
np.save('weight_saved.npy', weight_saved)


# calculate euclidean dist btw weight matrix and vector
dist = LA.norm((mnist.train.images[11,:] * 256 - weight_saved), axis = 2)
imshow(dist)


# Converting 3D SOM to 2D image
map_matrix = np.zeros((map_w*img_x, map_h*img_y))
for i in range(map_w):
    for j in range(map_h):
        # Reshaping 768 weight vector to 28x28 matrix
        reshaped_weights = mnist_map.weights[i][j].reshape((img_x, img_y))
        # Assigning matrix to respective position of node in lattice
        map_matrix[i*img_x:i*img_x+img_x, j*img_y:((j*img_y)+img_y)] = reshaped_weights
        
# Showing Image
map_img = Image.fromarray(map_matrix.astype('uint8'))
map_img.format = 'JPG'
map_img.show()

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
from PIL import ImageDraw
import matplotlib.pyplot as plt
fig = plt.figure("Distance Matrix Animation")
ax1 = fig.add_subplot(1,1,1)
Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
mapp = []
for i in range(0, len(distances), 9):
#    source_img = Image.fromarray(np.uint8(distances[i]))
    g_i=ax1.imshow(distances[i] ,animated=True)
    ax1.set_title("distance matrix")
    mapp.append([g_i])
    
ani = animation.ArtistAnimation(fig, mapp, interval = 1000, blit = False, repeat_delay = 500)
ani.save(r'D:\SOM\som_distance_mnist.mp4', writer = writer)
plt.show()

fig2=plt.figure("total_error vs epoch")
plt.plot(total_error)
plt.xlabel('Epochs')
plt.ylabel('Total_error')
plt.savefig(r'D:\SOM' + '\som_distance_mnist_100.png')

"""
Visualize terrain data
"""

import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
# Load the terrain
terrain1 = imread('data/saharaish.tif')
# Show the terrain
plt.figure()
plt.title('Terrain over Norway 1')
plt.imshow(terrain1, cmap='gray')
plt.colorbar()
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
# plt.savefig('plots/realdata/SRTM1.png')

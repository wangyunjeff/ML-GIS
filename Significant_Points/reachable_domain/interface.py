import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pandas as pd

import model.DEMclass as dem
import model.reachable_function as rf


def reachable_(dem_path):

    # Read Data
    img = Image.open(dem_path)
    npgrid = np.array(img)
    pre = npgrid
    npgrid = dem.AddRound(npgrid)

    # Compute the gradient in the x and y directions
    dx, dy = dem.Cacdxdy(npgrid, 22.5, 22.5)

    # Compute and save slope and aspect
    slope, arf = dem.CacSlopAsp(dx, dy)
    dem.np.savetxt("slope.csv", slope, delimiter=",", fmt='%f')

    # Compute velocity matrix
    v = rf.compute_velocity(slope)

    # Compute path and time matrix
    Ts = np.zeros(v.shape)
    for i in range(Ts.shape[1]):
        for j in range(Ts.shape[0]):
            paths = rf.get_path(rf.Point(100, 100), rf.Point(i, j))
            T = rf.get_T(v, paths)
            Ts[j, i] = T

    # Draw isochronous lines
    path = rf.draw_isochronous_lines(Ts)
    return path

if __name__ == '__main__':
    reachable_('./datasets/柞水.tif')
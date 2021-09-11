import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


def open_image(path):
    img = Image.open(dem_path)
    img_array = np.array(img)
    return img, img_array


def plot_array(array, show=True):
    fig, ax = plt.subplots()
    ax.imshow(array)
    plt.colorbar()
    ax.axis('off')
    if show == True:
        plt.show()

def plot_contour(array, scale=5):

    x_plt = np.arange(0, array.shape[0], 1)
    y_plt = np.arange(0, array.shape[1], 1)
    # X,Y = np.meshgrid(x,y)
    fig, ax = plt.subplots()
    # plt.contourf(x_plt, y_plt, data_array, 250, cmap='gist_earth')
    # ax.contourf(y_plt, x_plt, array, scale, cmap='gist_earth')
    plt.contourf(y_plt, x_plt, array, scale, cmap='gray_r')
    plt.colorbar()
    plt.show()

def plot_3d():
    # -*- coding: gbk -*-
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cbook
    from matplotlib import cm
    from matplotlib.colors import LightSource
    import matplotlib.pyplot as plt
    import numpy as np
    from osgeo import gdal

    gdal.AllRegister()

    filePath = './data/柞水.tif'  # 输入你的dem数据

    dataset = gdal.Open(filePath)
    adfGeoTransform = dataset.GetGeoTransform()
    band = dataset.GetRasterBand(1)  # 用gdal去读写你的数据，当然dem只有一个波段

    nrows = dataset.RasterXSize
    ncols = dataset.RasterYSize  # 这两个行就是读取数据的行列数

    Xmin = adfGeoTransform[0]  # 你的数据的平面四至
    Ymin = adfGeoTransform[3]
    Xmax = adfGeoTransform[0] + nrows * adfGeoTransform[1] + ncols * adfGeoTransform[2]
    Ymax = adfGeoTransform[3] + nrows * adfGeoTransform[4] + ncols * adfGeoTransform[5]

    x = np.linspace(Xmin, Xmax, ncols)
    y = np.linspace(Ymin, Ymax, nrows)
    X, Y = np.meshgrid(x, y)
    Z = band.ReadAsArray(0, 0, nrows, ncols)  # 这一段就是讲数据的x，y，z化作numpy矩阵

    region = np.s_[10:100, 10:101]
    X, Y, Z = X[region], Y[region], Z[region]

    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'), figsize=(12, 10))
    ls = LightSource(270, 20)  # 设置你可视化数据的色带
    rgb = ls.shade(Z, cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=rgb,
                           linewidth=0, antialiased=False, shade=False)

    plt.show()  # 最后渲染出你好看的三维图吧

if __name__ == '__main__':
    # dem_path = './data/柞水.tif'
    # img, img_array = open_image(dem_path)
    # plot_contour(img_array)
    #
    #
    # import pandas as pd
    # df = pd.read_csv('./slope.csv')
    # df_array = np.array(df)
    # # plot_array(df_array)
    # plot_contour(df_array,30)

    # plot_3d()
    import pandas as pd
    df = pd.read_csv('./slope.csv')
    df_array = np.array(df)+0.01
    # df_array = df_array + np.(df_array)

    # dem_path = './data/柞水.tif'
    # img, df_array = open_image(dem_path)

    v_array = 1/df_array
    # plt.imshow(v_array)
    # plt.show()
    v_array = v_array/np.max(v_array)
    np.savetxt('v.txt', v_array, fmt = '%f')

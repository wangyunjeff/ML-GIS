import DEMclass as dem
from DEMclass import Drawgrid
import numpy as np

####程序入口
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from PIL import Image
    import numpy as np

    v = np.loadtxt('./v.txt')
    plt.imshow(v, cmap='gray_r')
    plt.show()
    #
    dem_path = './datasets/柞水.tif'
    img = Image.open(dem_path)
    npgrid = np.array(img)

    # npgrid=dem.readfile()
    pre = npgrid
    npgrid = dem.AddRound(npgrid)
    dx, dy = dem.Cacdxdy(npgrid, 22.5, 22.5)
    slope, arf = dem.CacSlopAsp(dx, dy)
    dem.np.savetxt("slope2.csv", slope, delimiter=",",fmt = '%f')
    # # 绘制三维DEM
    Drawgrid(judge=1, pre=pre)
    # # 绘制二维DEM
    Drawgrid(judge=0, A=pre, strs="bone")
    # # 绘制坡度图
    Drawgrid(judge=0, A=slope, strs="rainbow")
    # # 绘制坡向图
    Drawgrid(judge=0, A=arf)

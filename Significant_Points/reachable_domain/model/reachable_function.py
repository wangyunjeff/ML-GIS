import numpy as np
import cv2
import tools as tl
import matplotlib.pyplot as plt


class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def getX(self):
        return self.x

    def getY(self):
        return self.y

# # 计算两点所走路径
# def get_path(currentPoint, tmpPoint):
#     paths = []
#     i, j = currentPoint.x, currentPoint.y
#     # paths.append([x,y])
#     for ii in range(tmpPoint.x-currentPoint.x+tmpPoint.y-currentPoint.y):
#         if i < tmpPoint.x:
#             paths.append([i,j])
#             i+=1
#         if j <= tmpPoint.y:
#             paths.append([i, j])
#             j += 1
#     return paths

# 计算两点所走路径---------
def get_path(currentPoint, tmpPoint):
    paths = []
    i, j = currentPoint.x, currentPoint.y
    # paths.append([x,y])
    # 第一象限
    if (tmpPoint.x > currentPoint.x) and (tmpPoint.y > currentPoint.y):
        for ii in range(abs(tmpPoint.x-currentPoint.x)+abs(tmpPoint.y-currentPoint.y)):
            if tmpPoint.x > i:
                paths.append([i,j])
                i += 1
            if tmpPoint.y > j:
                paths.append([i, j])
                j += 1
    # 第二象限
    elif (tmpPoint.x <= currentPoint.x) and (tmpPoint.y >= currentPoint.y):
        for ii in range(abs(currentPoint.x-tmpPoint.x) + abs(currentPoint.y-tmpPoint.y)):
            if tmpPoint.x < i:
                paths.append([i,j])
                i -= 1
            if tmpPoint.y > j:
                paths.append([i, j])
                j += 1
    # 第三象限
    elif (tmpPoint.x <= currentPoint.x) and (tmpPoint.y <= currentPoint.y):
        for ii in range(abs(currentPoint.x-tmpPoint.x) + abs(currentPoint.y-tmpPoint.y)):
            if tmpPoint.x < i:
                paths.append([i,j])
                i -= 1
            if tmpPoint.y < j:
                paths.append([i, j])
                j -= 1
    # 第四象限
    elif (tmpPoint.x >= currentPoint.x) and (tmpPoint.y <= currentPoint.y):
        for ii in range(abs(currentPoint.x-tmpPoint.x) + abs(currentPoint.y-tmpPoint.y)):
            if tmpPoint.x > i:
                paths.append([i,j])
                i += 1
            if tmpPoint.y < j:
                paths.append([i, j])
                j -= 1
    return paths

def compute_velocity(a):
    a = a + 0.01
    v_array = 1 / a
    v_array = v_array / np.max(v_array)

    return v_array
# 计算两点行走代价
# def get_cost(speed, paths):
#     speed_all = 0
#     for path in paths:
#         speed_all += speed[path[0],path[1]]
#     # speed_av = speed_all/len(paths)
#     return speed_all
#
#
# # 计算两点T
# def get_T(L, cost):
#     if cost == 0:
#         T = L / 0.00000001
#     else:
#         T = L / cost
#     return T

# 计算两点T
def get_T(speed, paths):
    # print(paths)
    T_all = 0
    for path in paths:
        T_all += 30/speed[path[1], path[0]]

    return T_all

def draw_isochronous_lines(Ts):
    save_path = './result/reachable.png'
    x_plt = np.arange(0, Ts.shape[1], 1)
    y_plt = np.arange(0, Ts.shape[0], 1)
    # X,Y = np.meshgrid(x,y)
    fig, ax = plt.subplots()
    # plt.contourf(x_plt, y_plt, data_array, 250, cmap='gist_earth')
    # ax.contourf(y_plt, x_plt, Ts, 10, cmap='gist_earth')

    # ax.yaxis.set_ticks_position('right')  # 将y轴的位置设置在右边
    ax.invert_yaxis()  # y轴反向
    plt.contourf(x_plt, y_plt, Ts, 10)
    plt.colorbar()
    plt.savefig(save_path)
    return save_path
# 计算两点距离
def getGrayDiff(currentPoint, tmpPoint):
    # return abs(int(img[currentPoint.x, currentPoint.y]) - int(img[tmpPoint.x, tmpPoint.y])
    return abs(int(np.sqrt(((currentPoint.x - tmpPoint.x) **2 + (currentPoint.y - tmpPoint.y) **2))))


def selectConnects(p):
    if p != 0:
        connects = [Point(-1, -1), Point(0, -1), Point(1, -1), Point(1, 0), Point(1, 1), Point(0, 1), Point(-1, 1),
                    Point(-1, 0)]
    else:
        connects = [Point(0, -1), Point(1, 0), Point(0, 1), Point(-1, 0)]
    return connects


def regionGrow(img, seeds, thresh, p=1):
    height, weight = img.shape
    seedMark = np.zeros(img.shape)
    seedList = []
    for seed in seeds:
        seedList.append(seed)
    label = 250
    connects = selectConnects(p)
    i_num= 0
    while len(seedList) > 0:
        currentPoint = seedList.pop(0)

        seedMark[currentPoint.x, currentPoint.y] = label


        for i in range(8):

            tmpX = currentPoint.x + connects[i].x
            tmpY = currentPoint.y + connects[i].y
            if tmpX < 0 or tmpY < 0 or tmpX >= height or tmpY >= weight:
                continue
            grayDiff = getGrayDiff(img, currentPoint, Point(tmpX, tmpY))
            if grayDiff < thresh and seedMark[tmpX, tmpY] == 0:
                seedMark[tmpX, tmpY] = label
                seedList.append(Point(tmpX, tmpY))
        i_num+=1
        if i_num % 50 == 0:
            plt.figure()
            # plt.savefig('./result/{}.jpg'.format(currentPoint.x))
            # plt.clf()
            plt.imshow(seedMark,'gray_r')
            # plt.pause(0.1)
            # plt.ioff()
        # plt.imshow(seedMark, 'gray_r')
            plt.show()
        print(i_num)
    return seedMark

if __name__ == '__main__':
    paths = get_path(Point(60, 60), Point(100, 100))
    print(paths)


    v= np.loadtxt('./v.txt')
    # plt.imshow(v, cmap='gray')
    # plt.show()
    # paths = get_path(Point(116, 90), Point(120, 100))
    # cost = get_cost(v, paths)
    # L = len(paths) * 30
    # T = get_T(L, cost)

    Ts = np.zeros(v.shape)
    # i
    for i in range(Ts.shape[1]):
        for j in range(Ts.shape[0]):

            paths = get_path(Point(100, 100), Point(i, j))
            T = get_T(v, paths)
            # try:
            #     T = get_T(v, paths)
            # except:
            #     print(paths)
            Ts[j, i] = T
            # print(j,i)

    plt.imshow(Ts, cmap='gray')
    plt.show()
    x_plt = np.arange(0, Ts.shape[1], 1)
    y_plt = np.arange(0, Ts.shape[0], 1)
    # X,Y = np.meshgrid(x,y)
    fig, ax = plt.subplots()
    # plt.contourf(x_plt, y_plt, data_array, 250, cmap='gist_earth')
    # ax.contourf(y_plt, x_plt, Ts, 10, cmap='gist_earth')

    # ax.yaxis.set_ticks_position('right')  # 将y轴的位置设置在右边
    ax.invert_yaxis()  # y轴反向
    plt.contourf(x_plt, y_plt, Ts, 10)
    plt.colorbar()
    plt.show()
    pass
    # filename = './data/青木川dem.tif'
    # im_data, im_geotrans, im_width, im_height = tl.read_img(filename)
    # seeds = [Point(300, 400)]
    # binaryImg = regionGrow(im_data, seeds, 167)
    #
    # plt.figure()
    # # plt.imshow(im_data, 'gray')
    # # plt.show()
    # plt.imshow(binaryImg, 'gray')
    # plt.show()
    # # cv2.imshow('image', binaryImg)
    # # cv2.waitKey(0)

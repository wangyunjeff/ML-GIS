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


# 计算高度差
def getGrayDiff(img, currentPoint, tmpPoint):
    return abs(int(img[currentPoint.x, currentPoint.y]) - int(img[tmpPoint.x, tmpPoint.y]))


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
    label = 255
    connects = selectConnects(p)
    i=0
    while len(seedList) > 0:
        currentPoint = seedList.pop(0)

        seedMark[currentPoint.x, currentPoint.y] = label
        i+=1
        if i // 10 == 1:
            plt.figure()
            plt.imshow(seedMark, 'gray_r')
            plt.show()

        for i in range(8):
            tmpX = currentPoint.x + connects[i].x
            tmpY = currentPoint.y + connects[i].y
            if tmpX < 0 or tmpY < 0 or tmpX >= height or tmpY >= weight:
                continue
            grayDiff = getGrayDiff(img, currentPoint, Point(tmpX, tmpY))
            if grayDiff < thresh and seedMark[tmpX, tmpY] == 0:
                seedMark[tmpX, tmpY] = label
                seedList.append(Point(tmpX, tmpY))
    return seedMark

if __name__ == '__main__':

    filename = './data/青木川dem.tif'
    im_data, im_geotrans, im_width, im_height = tl.read_img(filename)
    seeds = [Point(400, 300)]
    binaryImg = regionGrow(im_data, seeds, 3)

    plt.figure()
    plt.imshow(im_data, 'gray')
    plt.show()
    plt.imshow(binaryImg, 'gray')
    plt.show()
    cv2.imshow('image', binaryImg)
    cv2.waitKey(0)

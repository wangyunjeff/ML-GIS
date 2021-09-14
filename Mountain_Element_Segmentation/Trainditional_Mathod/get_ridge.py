import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import matplotlib
from skimage.filters import meijering, sato, frangi, hessian
from skimage.morphology import convex_hull_image
from skimage import color, morphology
from labelme import utils
import os

# from skimage import draw
# a, b = draw.line(100,100,150,50)

# The original image is inverted as the object must be white.

matplotlib.use("Qt5Agg")

root = './Data'
dem_root = os.path.join(root, 'dem')
remote_root = os.path.join(root, 'remote')
mask_root = os.path.join(root, 'mask')
maskwr_root = os.path.join(root, 'mask_with_remote')
maskwd_root = os.path.join(root, 'mask_with_dem')

file_names = []
As = os.listdir(remote_root)

for A in As:
    file_name = A.split('.')[0]
    file_names.append(file_name)

for file_name in file_names:
    dem_file = os.path.join(dem_root, file_name + '_dem.tif')
    remote_file = os.path.join(remote_root, file_name + '.jpg')

    remote_img = Image.open(remote_file)
    dem_img = Image.open(dem_file)
    remote_data = np.array(remote_img)
    dem_data = np.array(dem_img)

    # plt.imshow(remote_data)
    # plt.show()
    dem_img = Image.fromarray((dem_data - np.min(dem_data)) / (np.max(dem_data) - np.min(dem_data))*255).convert("RGB")
    # img.show()
    # plt.savefig(os.path.join(root, 'tem', '1.jpg'))

    result = hessian((dem_data-np.min(dem_data)) / (np.max(dem_data)-np.min(dem_data)),
                     sigmas=range(200, 500, 500), scale_range=None, scale_step=None,
                     alpha=0.5, beta=0.5, gamma=15, black_ridges=False, mode='reflect',
                     cval=0)
    res = morphology.white_tophat(result)
    result = result - res
    # result = meijering((dem_data-np.min(dem_data)) / (np.max(dem_data)-np.min(dem_data)),
    #                    sigmas=range(50, 300, 100), black_ridges=True)
    image_with_remote = Image.blend(Image.fromarray(np.uint8(remote_data)), Image.fromarray(np.uint8(result*255)).convert("RGB"), 0.3)
    # image.show()
    image_with_remote.save(os.path.join(maskwr_root, file_name+'.png'))

    image_with_dem = Image.blend(dem_img, Image.fromarray(np.uint8(result*255)).convert("RGB"), 0.1)
    image_with_dem.save(os.path.join(maskwd_root, file_name + '.png'))
    utils.lblsave(os.path.join(mask_root, file_name) + '.png', result)
    # plt.imshow(image, cmap='gray')

    print(file_name)

file_path = './Data/baicha_dem.tif'




# range(3, 66, 30) 分别控制：线条起始粗细、线条筛选阈值(线条多少)、线条精细度
# (3, 30, 3)在90m精度有较好结果
# result = hessian((tif_array-np.min(tif_array)) / (np.max(tif_array)-np.min(tif_array)), sigmas=range(50, 300, 100), scale_range=None, scale_step=None,
#                  alpha=0.5, beta=0.5, gamma=15, black_ridges=False, mode=None,
#                  cval=0)
# result = meijering((tif_array-np.min(tif_array)) / (np.max(tif_array)-np.min(tif_array)), sigmas=range(50, 300, 100),black_ridges=False)
# plt.imshow(result, cmap='gray')
# plt.show()
# utils.lblsave(save_root_png + '/' + file.split('.')[0] + '.png', result)


# plt.imshow(tif10_array[2000:2473,2000:2473]/7800)
# plt.show()
# plt.imshow(result)
# fig, ax = plt.subplots(1,2,figsize=(8,16),dpi = 50)
# fig, ax = plt.subplots(1, 1, figsize=(12.01, 12.01), dpi=100)
# ax[0].imshow(tif90_array)
# ax[1].imshow(tif10_array)
# plt.show()

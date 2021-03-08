import os
import scipy.io
from skimage import io
from scipy import ndimage
import cv2
from tqdm import tqdm


def create_dir_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)


# mat = scipy.io.loadmat('/home/mvries/Documents/Datasets/Pix3D/model/chair/IKEA_BERNHARD/voxel.mat')
# io.imsave('/home/mvries/Documents/Datasets/Pix3D/model/chair/IKEA_BERNHARD/voxel.tif', mat)


def pix_3d_dataset(directory, save_directory, re_size):
    create_dir_if_not_exist(save_directory)
    image_total = 0
    cat_listed_dir = sorted(os.listdir(directory))
    if '.DS_Store' in cat_listed_dir:
        cat_listed_dir.remove(".DS_Store")

    for category in tqdm(cat_listed_dir):
        cat_folder_dir = directory + category + '/'
        listed_items_in_cats = sorted(os.listdir(cat_folder_dir))
        for i, item_in_cat in enumerate(listed_items_in_cats):
            item_cat_dir = cat_folder_dir + item_in_cat + '/'
            if os.path.isfile(item_cat_dir + 'voxel.mat'):
                mat = scipy.io.loadmat(item_cat_dir + 'voxel.mat')
                norm_image = cv2.normalize(mat['voxel'],
                                           None,
                                           alpha=0,
                                           beta=255,
                                           norm_type=cv2.NORM_MINMAX)

                z, x, y = norm_image.shape[0], norm_image.shape[1], norm_image.shape[2]

                if re_size is not None:
                    zoom = (re_size[0] / z, re_size[1] / x, re_size[2] / y)
                    # print(zoom)
                    img = ndimage.zoom(norm_image, zoom)
                else:
                    img = norm_image

                save_to = save_directory + category + '/'
                create_dir_if_not_exist(save_to)
                io.imsave(save_to + str(i).zfill(3) + '.tif', img)
                image_total += 1

    print(image_total)


if __name__ == '__main__':
    dir = '/home/mvries/Documents/Datasets/Pix3D/model/'
    save_dir = '/home/mvries/Documents/Datasets/Pix3DVoxels/'
    resize = [64, 64, 64]
    pix_3d_dataset(dir, save_dir, resize)

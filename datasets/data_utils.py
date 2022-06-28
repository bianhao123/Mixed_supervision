from albumentations.pytorch import ToTensorV2
import albumentations as A
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tifffile as tiff


MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])


def visulize(input_img, mode='opencv'):
    if mode == 'opencv':
        plt.figure(figsize=(15,10))
        plt.imshow(cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB))
        plt.show()

def get_transforms_test():
    transforms = A.Compose([
        A.Normalize(mean=(MEAN[0], MEAN[1], MEAN[2]),
                  std=(STD[0], STD[1], STD[2])),
        ToTensorV2(),
    ])
    return transforms


def denormalize(z, mean=MEAN.reshape(-1, 1, 1), std=STD.reshape(-1, 1, 1)):
    return std*z + mean


def draw_strcuture_from_hue(image, fill=255, scale=1/32):

    height, width, _ = image.shape
    vv = cv2.resize(image, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    vv = cv2.cvtColor(vv, cv2.COLOR_RGB2HSV)
    # image_show('v[0]', v[:,:,0])
    # image_show('v[1]', v[:,:,1])
    # image_show('v[2]', v[:,:,2])
    # cv2.waitKey(0)
    mask = (vv[:, :, 1] > 32).astype(np.uint8) #相当于做一个过滤，把小于32的去除
    mask = mask*fill
    mask = cv2.resize(mask, dsize=(width, height), interpolation=cv2.INTER_LINEAR)

    return mask


def to_tile(image, mask, structure, scale, size, step, min_score):

    half = size//2 #image类型numpy
    image_small = cv2.resize(image, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    height, width, _ = image_small.shape

    #make score
    structure_small = cv2.resize(structure, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    vv = structure_small.astype(np.float32)/255


    #make coord
    xx = np.linspace(half, width  - half, int(np.ceil((width  - size) / step)))
    yy = np.linspace(half, height - half, int(np.ceil((height - size) / step)))
    xx = [int(x) for x in xx]
    yy = [int(y) for y in yy]

    coord  = []
    reject = []
    for cy in yy:
        for cx in xx:
            cv = vv[cy - half:cy + half, cx - half:cx + half].mean()
            if cv>min_score:
                coord.append([cx,cy,cv]) #去除背景元素
            else:
                reject.append([cx,cy,cv])
    #-----
    if 1:
        tile_image = []
        for cx,cy,cv in coord:
            t = image_small[cy - half:cy + half, cx - half:cx + half]
            assert (t.shape == (size, size, 3))
            tile_image.append(t)

    if mask is not None:
        mask_small = cv2.resize(mask, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        tile_mask = []
        for cx,cy,cv in coord:
            t = mask_small[cy - half:cy + half, cx - half:cx + half]
            assert (t.shape == (size, size))
            tile_mask.append(t)
    else:
        mask_small = None
        tile_mask  = None

    return {
        'image_small': image_small,
        'mask_small' : mask_small,
        'structure_small' : structure_small,
        'tile_image' : tile_image,
        'tile_mask'  : tile_mask,
        'coord'  : coord,
        'reject' : reject,
    }

def read_tiff(image_file: str) -> np.ndarray:
    """read tiff

    Args:
        image_file (str): [description]

    Returns:
        ndarray: return HWC ndarray image
    """
    image = tiff.imread(image_file)
    if image.shape[:2] == (1, 1):
        image = image.squeeze(0).squeeze(0)
    if image.shape[0] == 3:
        image = image.transpose(1, 2, 0)
        image = np.ascontiguousarray(image)

    return image
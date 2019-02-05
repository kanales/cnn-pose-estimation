import numpy as np
import re
import os
import random
import matplotlib.image as mpimg

from collections import namedtuple
from itertools import count

from multiprocessing import Pool

__all__ = ["normalize", "read_images"]

Sample = namedtuple('Sample', ['cls', 'idx','quat','img'])
def normalize(mat):
    """
    Normalizes np.array to mean 0 and var 1
    """
    out = np.empty_like(mat)
    for i in range(3):
        m = np.mean(mat[:,:,i])
        v = np.var(mat[:,:,i])
        out[:,:,i] = (mat[:,:,i] - m) / v
    return out

def un_normalize(img):
    img = img - img.min()
    img = img * 255 / img.max()
    return img.astype(int)

def _processer(imname, pattern = re.compile(r'.*[A-Za-z]+([0-9]+)\.png')):
    m = pattern.match(imname)
    if m:
        return int(m.group(1)), normalize(mpimg.imread(imname))
    else:
        return None, None

def _read_folder(folder, cls):
    """
    Helper function, iterates through a folder reading samples
    """
    # quats
    quats = []
    with open(os.path.join(folder,'poses.txt'),'r') as f:
        while True:
            line1 = f.readline()
            line2 = f.readline()
            if not line2: break
            quats.append(np.array([float(x) for x in line2.split()]))

    # images
    imgs = [None] * len(quats)

    pool = Pool()
    
    for i, img in pool.map(_processer, [os.path.join(folder,imname) for imname in os.listdir(folder)]):
        if i and img is not None:
            imgs[i] = img
    pool.close()
    pool.join()

    # for imname in os.listdir(folder):
    #     m = pattern.match(imname)
    #     if m:
    #         imgs[int(m.group(1))] = normalize(mpimg.imread(os.path.join(folder,imname)))
            
    return [Sample(cls, idx, quat, img) for idx, quat, img in zip(count(),quats,imgs)]

def read_images(path):
    """
    Returns a list of samples with all the images in `path`
    """
    data = []
    for root, subfolders,_ in os.walk(path):
        for i,folder in enumerate(sorted(subfolders)):
            data.append(_read_folder(os.path.join(root,folder), i))

    return [l for lst in data for l in lst]
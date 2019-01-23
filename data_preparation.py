import numpy as np
import re
import os
import random
import matplotlib.image as mpimg

from collections import namedtuple
from itertools import count

__all__ = ["normalize", "read_images", "similarity", "batch"]

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

def _read_folder(folder, cls, pattern = re.compile(r'[A-Za-z]+[0-9]+\.png')):
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
    imgs = []
    
    for imname in os.listdir(folder):
        if pattern.match(imname):
            imgs.append(
                normalize(mpimg.imread(os.path.join(folder,imname))))
            
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

def similarity(q1, q2):
    """
    Returns a measure of similarity between two quaternions
    """
    return 2 * np.arccos(min(1,np.abs(q1 @ q2)))

def batch(Sdb, Strain, n):
    """
    Generates a batch of `n` elements
    """
    def gen():
        for x in range(n):
            # Anchor: select random sample from Strain
            anchor = random.choice(Strain)
            # Puller: select most similar from Sdb
            puller = max(Sdb, key = lambda x: similarity(x.quat,anchor.quat))
            # Pusher: same object different pose | random different object 
            pusher = random.choice([x for x in Sdb if x.cls != anchor.cls])
            yield anchor.img
            yield puller.img
            yield pusher.img
    return np.array(list(gen()))
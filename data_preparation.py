import numpy as np
import re
import os
import random
import matplotlib.image as mpimg

from collections import namedtuple
from itertools import count

Sample = namedtuple('Sample', ['cls', 'idx','quat','img'])
def normalize(mat):
    out = np.empty_like(mat)
    for i in range(3):
        m = np.mean(mat[:,:,i])
        v = np.var(mat[:,:,i])
        out[:,:,i] = (mat[:,:,i] - m) / v
    return out

def parse(folder, cls):
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
    p = re.compile(r'[A-Za-z]+[0-9]+\.png')
    for imname in os.listdir(folder):
        if p.match(imname):
            imgs.append(
                normalize(mpimg.imread(os.path.join(folder,imname))))
            
    return [Sample(cls, idx, quat, img) for idx, quat, img in zip(count(),quats,imgs)]

def read_images(path):
    data = []
    for root, subfolders,_ in os.walk(path):
        for i,folder in enumerate(sorted(subfolders)):
            data.append(parse(os.path.join(root,folder), i))

    return [l for lst in data for l in lst]

def similarity(q1, q2):
    return 2 * np.arccos(min(1,np.abs(q1 @ q2)))

def batch(Sdb, Strain, n):
    def gen(m):
        for x in range(m):
            # Anchor: select random sample from Strain
            anchor = random.choice(Strain)
            # Puller: select most similar from Sdb
            puller = max(Sdb, key = lambda x: similarity(x.quat,anchor.quat))
            # Pusher: same object different pose | random different object 
            pusher = random.choice([x for x in Sdb if x.cls != anchor.cls])
            yield anchor
            yield puller
            yield pusher
    return list(gen(n))
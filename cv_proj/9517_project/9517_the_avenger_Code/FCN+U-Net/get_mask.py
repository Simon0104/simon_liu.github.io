import torch
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

def getClassName(classID, cats):
    for i in range(len(cats)):
        if cats[i]['id']==classID:
            return cats[i]['name']
    return "None"


def get_mask_with_id(id):
    annFile = "/turtles-data/data/annotations.json"
    coco = COCO(annFile)
    catIDs = coco.getCatIds()
    cats = coco.loadCats(catIDs)

    filterClasses = ['laptop', 'tv', 'cell phone']
    catIds = coco.getCatIds(catNms=filterClasses)
    imgIds = coco.getImgIds(catIds=catIds)

    img = coco.loadImgs(imgIds[id])[0]

    annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)
    coco.showAnns(anns)

    filterClasses = ['turtle', 'flipper', 'head']
    mask = np.zeros((img['height'], img['width']))
    for i in range(len(anns)):
        className = getClassName(anns[i]['category_id'], cats)
        pixel_value = filterClasses.index(className) + 1
        mask = np.maximum(coco.annToMask(anns[i]) * pixel_value, mask)

    return torch.from_numpy(mask)
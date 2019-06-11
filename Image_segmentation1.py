
import sys
import os
import cv2
from mxnet import gluon, image, nd
from mxnet.gluon import data as gdata, utils as gutils
import d2l
import numpy as np
import tensorflow as tf
import sklearn
import mxnet

voc_dir = "C:\\Users\\bhatt\\Desktop\\Jupyter Notebook\\VOCdevkit\\VOC2012"

def read_voc_images(root=voc_dir, is_train=True):
    txt_fname = '%s/ImageSets/Segmentation/%s' % (
        root, 'train.txt' if is_train else 'val.txt')
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    features, labels = [None] * len(images), [None] * len(images)
    for i, fname in enumerate(images):
        features[i] = image.imread('%s/JPEGImages/%s.jpg' % (root, fname))
        labels[i] = image.imread(
            '%s/SegmentationClass/%s.png' % (root, fname))
    return features, labels

train_features, train_labels = read_voc_images()

n = 5
imgs = train_features[0:n] + train_labels[0:n]
d2l.show_images(imgs, 2, n);

import os
import torch
import torchvision
import torchvision.datasets as datasets


def return_somethingv1(ROOT_DATASET):
    filename_categories = 'somethingv1/category.txt'
    root_data = ROOT_DATASET + 'somethingv1/20bn-something-something-v1'
    filename_imglist_train = 'somethingv1/train.txt'
    filename_imglist_val = 'somethingv1/valid.txt'
    prefix = '{:05d}.jpg'
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_somethingv2(ROOT_DATASET):
    filename_categories =  '/home/shiyanshi/pingyi/project/TFM/videofolder/category.txt'
    root_data = ROOT_DATASET
    filename_imglist_train = '/home/shiyanshi/pingyi/dataset/smthv2/videofolder/train.txt'
    filename_imglist_val = '/home/shiyanshi/pingyi/project/TFM/videofolder/val_videofolder.txt'
    prefix = '{:06d}.jpg'
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_dataset(dataset,ROOT_DATASET):
    dict_single = { 'somethingv1':return_somethingv1, 'somethingv2':return_somethingv2}
    if dataset in dict_single:
        file_categories, file_imglist_train, file_imglist_val, root_data, prefix = dict_single[dataset](ROOT_DATASET)
    else:
        raise ValueError('Unknown dataset '+dataset)

    file_imglist_train = os.path.join(ROOT_DATASET, file_imglist_train)
    file_imglist_val = os.path.join(ROOT_DATASET, file_imglist_val)
    file_categories = os.path.join(ROOT_DATASET, file_categories)
    with open(file_categories) as f:
        lines = f.readlines()
    categories = [item.rstrip() for item in lines]
    return categories, file_imglist_train, file_imglist_val, root_data, prefix


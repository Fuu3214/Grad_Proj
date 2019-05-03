#coding=utf-8
import tensorflow as tf
import numpy as np
import os
import pandas as pd

# DATA_DIR = './splitted_dataset/adv_train'
# LABEL = './splitted_dataset/adv_train.csv'

DATA_DIR = '../train'
LABEL = '../train_labels.csv'

def make_dataset(
    data_dir = DATA_DIR, 
    label_dir = LABEL,
    batch_size = 32
):
    img_files = []

    for root, dirs, files in os.walk(data_dir):
        for fileName in files:
            img_files.append(data_dir + os.sep + fileName)

    img_files.sort(key=lambda x:int(x[len('../train/train_'):-4]))
    labels_lst = pd.read_csv(label_dir, header = None).rename(columns = {0:'label'})['label'].values.tolist()
    def _parse_function(filename, label):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_bmp(image_string)
        image_resized = tf.reshape(image_decoded, shape = [784])/255
        return image_resized, label

    # 图片文件的列表
    filenames = tf.constant(img_files)
    # label[i]就是图片filenames[i]的label
    labels = tf.constant(labels_lst)

    # 此时dataset中的一个元素是(filename, label)
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))

    # 此时dataset中的一个元素是(image_resized, label)
    dataset = dataset.map(_parse_function)

    # 此时dataset中的一个元素是(image_resized_batch, label_batch)
    dataset = dataset.shuffle(buffer_size=2048).repeat().batch(batch_size)
    
    return dataset


#!/usr/bin/env python
# coding: utf-8

# In[10]:



import os
import tensorflow as tf
from PIL import Image  #注意Image,后面会用到
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import struct


# In[11]:



DATA_DIR = './splitted_dataset/adv_train'
LABEL = './splitted_dataset/adv_train.csv'

writer= tf.python_io.TFRecordWriter("adv_train.tfrecords") #要生成的文件
 

imagepaths = []
labels = []

for root, dirs, files in os.walk(DATA_DIR):
    for fileName in files:
        imagepaths.append(DATA_DIR + os.sep + fileName)

imagepaths.sort(key=lambda x:int(x[len(DATA_DIR)+len(os.sep):-4]))
labels = pd.read_csv(LABEL, header = None).rename(columns = {0:'label'})['label'].values.tolist()

for img_path,label in zip(imagepaths, labels):
    img_raw = Image.open(img_path).convert('L')  
    img_raw = img_raw.resize((28, 28))     # 转换图片大小
    img_raw = img_raw.tobytes()       # 将图片转化为原生bytes
    example = tf.train.Example(features=tf.train.Features(feature={
        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
        'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
    })) #example对象对label和image数据进行封装
    writer.write(example.SerializeToString())  #序列化为字符串
 
writer.close()


# In[ ]:





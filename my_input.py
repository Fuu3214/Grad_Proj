#coding=utf-8
import tensorflow as tf

def read_and_decode(filename_queue, batch_size):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
      serialized_example,
      features={             
          'img_raw': tf.FixedLenFeature([],tf.string),
          'label': tf.FixedLenFeature([], tf.int64),  
      })

    image = tf.decode_raw(features['img_raw'],tf.uint8)
    image = tf.reshape(image, shape = [784])
    image = tf.cast(image, tf.float32) * (1. / 255)
    label = features['label']

    images, labels = tf.train.shuffle_batch([image, label],
                                min_after_dequeue=1000,
                                batch_size=batch_size,
                                capacity=2000,
                                num_threads=4)
    return images, labels
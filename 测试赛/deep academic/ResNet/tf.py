from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

minst = input_data.read_data_sets('C:\\Users\\ouguangji\\Desktop\\图片数据\\MNIST_data', one_hot=True)
print(minst)
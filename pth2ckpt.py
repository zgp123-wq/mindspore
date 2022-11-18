import tensorflow.compat.v1 as tf
import torch

def convert(bin_path, ckptpath):
    with  tf.Session() as sess:
        for var_name, value in torch.load(bin_path, map_location='cpu').items():
            print(var_name)  # 输出权重文件中的变量名
            tf.Variable(initial_value=value, name=var_name)
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        saver.save(sess, ckpt_path)

bin_path = '/home/data1/zgp/paa/pre_train/resnet50-19c8e357.pth'
ckpt_path = '/home/data1/zgp/paa/pre_train/resnet50.ckpt'
convert(bin_path, ckpt_path)
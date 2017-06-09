
import tensorflow as tf
## 添加一个神经层
def add_layer(inputs,in_size,out_size,activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))
    biase = tf.Variable(tf.zeros([1,out_size])) + 0.1
    Wx_plus_b = tf.matmul(input,Weights) + biase
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs
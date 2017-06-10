
import tensorflow as tf
import  numpy as np
import  matplotlib.pyplot as  plt
## 1. 添加一个神经层
## 2. 结果可视化


def add_layer(inputs,in_size,out_size,n_layer,activation_function=None):
    # 添加一 或者多层 返回输出层
    layer_name = 'layer%s' % n_layer
    with tf.name_scope('layer'):
        with tf.name_scope('weghts'):
            Weights = tf.Variable(tf.random_normal([in_size,out_size]),name='W')
            tf.summary.histogram(layer_name+'/weghts',Weights)
        with tf.name_scope('biase'):
            biase = tf.Variable(tf.zeros([1,out_size]),name='Y') + 0.1
            tf.summary.histogram(layer_name+'/biase',biase)
        with tf.name_scope('wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs,Weights) + biase
            if activation_function is None:
                outputs = Wx_plus_b
            else:
                outputs = activation_function(Wx_plus_b)
            tf.summary.histogram(layer_name+'/outputs',outputs)
            return outputs
## 构造真实数据
x_data = np.linspace(-1,1,300)[:,np.newaxis]
noise = np.random.normal(0,0.05,x_data.shape)
y_date = np.square(x_data) - 0.5 + noise

# define placeholder for input
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32,[None,1],name='x_input')
    ys = tf.placeholder(tf.float32,[None,1],name='y_input')

l1 = add_layer(xs,1,10,n_layer=1,activation_function=tf.nn.relu)
predition = add_layer(l1,10,1,n_layer=2,activation_function=None)

with tf.name_scope('loss'):
    loss =tf.reduce_mean(tf.reduce_sum( tf.square(ys - predition),
                     reduction_indices=[1]))
    tf.summary.scalar('loss',loss)

with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

sess = tf.Session()

merged = tf.summary.merge_all()

if int((tf.__version__).split('.')[1])<12 and int((tf.__version__).split('.')[0])<1:# tensorflow version < 0.12
    writer = tf.train.SummaryWriter("logs/",sess.graph)
else:# tensorflow version >= 0.12
    writer = tf.summary.FileWriter("logs/",sess.graph)

# if using tensorflow >=0.12
if int((tf.__version__).split('.')[1])<12 and int((tf.__version__).split('.')[0])<1:# tensorflow version < 0.12
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()



## 收集框架信息

sess.run(init)

for i in range(1000):
    sess.run(train_step,feed_dict={xs:x_data,ys:y_date})
    if i% 50 ==0:
        result = sess.run(merged,feed_dict={xs: x_data,ys:y_date})
        writer.add_summary(result,i)

# 注释画图信息
# fig = plt.figure()
# ax = fig.add_subplot(1,1,1)
# ax.scatter(x_data,y_date)
# plt.ion()
# plt.show()
#
# for i in range(1000):
#     sess.run(train_step,feed_dict={xs:x_data,ys:y_date})
#     if i % 50 == 0:
#         # print(sess.run(loss,feed_dict={xs:x_data,ys:y_date}))
#         try:
#             ax.lines.remove(lines[0])
#         except Exception:
#             pass
#         predition_value= sess.run(predition,feed_dict={xs:x_data})
#         lines = ax.plot(x_data,predition_value,'r-',lw=5)
#         plt.pause(0.1)

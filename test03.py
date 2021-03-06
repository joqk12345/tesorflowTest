import tensorflow as tf

## session 管理

matrix1 = tf.constant([[3,3]])
matrix2 = tf.constant([[2],
                       [2]])

product = tf.matmul(matrix1,matrix2)

sess = tf.Session()
result = sess.run(product)
print(result)
sess.close()

## method2
with tf.Session() as sess:
    result2 = sess.run(product)
    print(result2)
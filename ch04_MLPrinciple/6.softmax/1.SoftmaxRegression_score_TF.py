import tensorflow as tf
import numpy as np

def softmax_regression_1():
    # y = x
    x = [[1., 2.],  # C
         [2., 1.],
         [4., 5.],  # B
         [5., 4.],
         [8., 9.],  # A
         [9., 8.]]
    y = [[0., 0., 1.],
         [0., 0., 1.],
         [0., 1., 0.],
         [0., 1., 0.],
         [1., 0., 0.],
         [1., 0., 0.]]

    w = tf.Variable(tf.random_normal([2, 3]))
    b = tf.Variable(tf.random_normal([3]))

    # (6, 3) = (6, 2) @ (2, 3)
	# z = tf.matmul(x, w) + b
    z = x @ w + b
    hx = tf.nn.softmax(z)

    loss_i = tf.nn.softmax_cross_entropy_with_logits_v2(logits=z, labels=y)
    loss = tf.reduce_mean(loss_i)

    optimizeer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizeer.minimize(loss)

    sess = tf.Session()
    sess.run((tf.global_variables_initializer()))

    for i in range(10):
        sess.run(train)
        print(i, sess.run(loss))

    preds = sess.run(hx)
    print(preds)
    sess.close()


def softmax_regression_2():
    x = [[1., 2.],  # C
         [2., 1.],
         [4., 5.],  # B
         [5., 4.],
         [8., 9.],  # A
         [9., 8.]]
    y = [[0., 0., 1.],
         [0., 0., 1.],
         [0., 1., 0.],
         [0., 1., 0.],
         [1., 0., 0.],
         [1., 0., 0.]]

    w = tf.Variable(tf.random_normal([2, 3]))
    b = tf.Variable(tf.random_normal([3]))

    ph_x = tf.placeholder(tf.float32)

    # (6, 3) = (6, 2) @ (2, 3)
    z = ph_x @ w + b
    hx = tf.nn.softmax(z)

    loss_i = tf.nn.softmax_cross_entropy_with_logits_v2(logits=z,
                                                        labels=y)
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(100):
        sess.run(train, {ph_x: x})
        print(i, sess.run(loss, {ph_x: x}))

    preds = sess.run(hx, {ph_x: x})
    # preds = sess.run(z, {ph_x: x})  # 가능함
    print(preds)
    print(y)
    print('-'*50)

    pred_arg = np.argmax(preds, axis=1)
    y_arg = np.argmax(y, axis=1)
    print(pred_arg)
    print(y_arg)

    print(np.mean(pred_arg == y_arg))

    grades = np.array(['A', 'B', 'C'])
    print(grades[pred_arg])
    print('-'*50)

    # 3시간 공부하고 7번 출석한 학생과
    # 6시간 공부하고 2번 출석한 학생의 학점을 알려주세요.
    preds = sess.run(z, {ph_x: [[3., 7.], [6., 2.]]})
    print(preds)
    pred_arg = np.argmax(preds, axis=1)
    print(pred_arg)

    sess.close()



# softmax_regression_1()
softmax_regression_2()

import tensorflow as tf
import numpy as np

def get_iris():
    iris = np.loadtxt('data/iris.csv', delimiter=',', dtype=np.float32)
    print(iris)
    print(iris.shape, iris.dtype)  # (150, 7) float64 --> float32로 변환

    np.random.shuffle(iris)

    x = iris[:, :4]
    y = iris[:, 4:]  # 1, 0, 0  One-hot vector
    print(x.shape, y.shape)  # (150, 4) (150, 3)

    train_size = int(len(x) * 0.7)
    x_train, x_test = x[:train_size], x[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    print(x_train.shape, x_test.shape)  # (105, 4) (45, 4)
    print(y_train.shape, y_test.shape)  # (105, 3) (45, 3)

    return x_train, y_train, x_test, y_test


# iris 시험 데이터 셋에 대해서 90% 이상의 정확도로 예측하세요.
def softmax_regression_iris():
    x_train, y_train, x_test, y_test = get_iris()

    #w = tf.Variable(tf.random_normal([4, 3])) # 4:x_train.shape[1], 3:y_train.shape[1]
    #b = tf.Variable(tf.random_normal([3])) # 3:y_train.shape[1]
    w = tf.Variable(tf.random_normal([x_train.shape[1], y_train.shape[1]])) # 4:x_train.shape[1], 3:y_train.shape[1]
    b = tf.Variable(tf.random_normal([y_train.shape[1]])) # 3:y_train.shape[1]

    ph_x = tf.placeholder(tf.float32)  # (105, 4) (x_train.shape[0], x_train.shape[1])

    # (105, 3) = (x_train.shape[0], x_train.shape[1]) @ (x_train.shape[1], y_train.shape[1])
	# (105, 3) = (105, 4) @ (4, 3)
    z = ph_x @ w + b
    hx = tf.nn.softmax(z)

    loss_i = tf.nn.softmax_cross_entropy_with_logits_v2(logits=z,
                                                        labels=y_train)
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(1000):
        sess.run(train, {ph_x: x_train})
        if i%10 == 0: print(i, sess.run(loss, {ph_x: x_train}))
    print('-' * 50)

    preds = sess.run(hx, {ph_x: x_test})
    # print(preds)
    pred_arg = np.argmax(preds, axis=1)
    print(pred_arg)

    y_arg = np.argmax(y_test, axis=1)
    print(y_arg)

    print('acc:', np.mean(pred_arg == y_arg))

    #types = np.array(['Setosa', 'Versicolor', 'Virginica'])
    #print(types[pred_arg])
    #print('-'*50)


    sess.close()

softmax_regression_iris()

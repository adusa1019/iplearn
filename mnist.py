#!/usr/bin/env python
# coding=utf-8

from chainer import Chain, Variable, optimizers
import chainer.functions
import cv2
# import dlib
import numpy
import os
import pickle
from sklearn.datasets import fetch_mldata
import sys
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def forward(x_batch, t_batch, model, is_train=True):
    x, t = Variable(x_batch), Variable(t_batch)
    h1 = chainer.functions.max_pooling_2d(chainer.functions.relu(model.conv1(x)), 3)
    h2 = chainer.functions.max_pooling_2d(chainer.functions.relu(model.conv2(h1)), 3)
    h3 = chainer.functions.dropout(chainer.functions.relu(model.l1(h2)), train=is_train)
    h4 = chainer.functions.dropout(chainer.functions.relu(model.l2(h3)), train=is_train)
    p = model.l3(h4)
    # 誤差を算出
    return chainer.functions.softmax_cross_entropy(p, t), chainer.functions.accuracy(p, t)


def predict(x_batch, model, is_train=True):
    x = Variable(x_batch)
    h1 = chainer.functions.max_pooling_2d(chainer.functions.relu(model.conv1(x)), 3)
    h2 = chainer.functions.max_pooling_2d(chainer.functions.relu(model.conv2(h1)), 3)
    h3 = chainer.functions.dropout(chainer.functions.relu(model.l1(h2)), train=is_train)
    h4 = chainer.functions.dropout(chainer.functions.relu(model.l2(h3)), train=is_train)
    p = model.l3(h4)
    return p.data


def train_detector(data, label, model_path='model',  model=None, optimizer=None, n_batch=20, n_epoch=20):
    num_data = len(label)
    if model is None:
        model = Chain(
            conv1=chainer.functions.Convolution2D(1, 32, 3),
            conv2=chainer.functions.Convolution2D(32, 64, 3),
            l1=chainer.functions.Linear(576, 200),
            l2=chainer.functions.Linear(200, 100),
            l3=chainer.functions.Linear(100, 10)
        )
    if optimizer is None:
        optimizer = optimizers.Adam()
    optimizer.setup(model)

    for epoch in range(n_epoch):
        print("epoch : %d" % (epoch + 1))

        # ランダムに並び替える
        perm = numpy.random.permutation(num_data)
        sum_accuracy = 0
        sum_loss = 0

        # バッチサイズごとに学習
        b_start = time.time()
        for i in range(0, num_data, n_batch):
            x_batch = data[perm[i:i + n_batch]]
            t_batch = label[perm[i:i + n_batch]]

            # 勾配を初期化
            optimizer.zero_grads()
            # 順伝搬
            loss, accuracy = forward(x_batch, t_batch, model)
            # 誤差逆伝搬
            loss.backward()
            optimizer.update()  # パラメータ更新

            sum_loss += float(loss.data) * n_batch
            sum_accuracy += float(accuracy.data) * n_batch

        # 誤差と精度を表示
        print("[学習]loss: %f, accuracy: %f, time: %f秒"
              % (sum_loss / num_data, sum_accuracy / num_data, time.time() - b_start))

    # 学習したモデルを保存
    pickle.dump(model, open(model_path, 'wb'), -1)


# test the detector
def test_detector(data, label, model_path='model'):
    if not os.path.exists(model_path):
        print('invalid model path')
        sys.exit()
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    p_start = time.time()
    results_score = predict(data, model, is_train=False)
    results = numpy.argmax(results_score, axis=1)

    predict_time = time.time() - p_start
    print("[認識] %f秒" % predict_time)
    # print("結果: " + str(results))
    # print("ラベル: " + str(label))
    print(classification_report(label, results))
    print(accuracy_score(label, results))
    print(confusion_matrix(label, results))


# detect the type of an input image
def detect_image(data,  model_path='model'):
    if not os.path.exists(model_path):
        print('invalid model path')
        sys.exit()
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    p_start = time.time()
    results_score = predict(data, model, is_train=False)
    results = numpy.argmax(results_score, axis=1)

    predict_time = time.time() - p_start
    print("[認識] %f秒" % predict_time)
    print("結果: " + str(results))
    print("score: " + str(results_score))
    pass


if __name__ == '__main__':
    if len(sys.argv) == 1:
        mnist = fetch_mldata('MNIST original')
        data_set = numpy.asarray(mnist.data, numpy.float32)
        labels = numpy.asarray(mnist.target, numpy.int32)
        data_set = data_set.reshape((len(data_set), 1, 28, 28))
        data_train, data_test, label_train, label_test = train_test_split(data_set, labels, test_size=0.2)
        print("train data = %d" % label_train.size)
        print("test data = %d" % label_test.size)
        # train_detector(data_train, label_train)
        test_detector(data_test, label_test)
    elif len(sys.argv) == 2:
        pass
        # mnist = fetch_mldata('MNIST original')
        # data_set = numpy.asarray(mnist.data[:100], numpy.float32)
        # print(len(data_set))
        # for data in data_set:
        #     data = numpy.reshape(data, (28, 28))
        #     cv2.imshow('data', cv2.resize(data, (560, 560)))
        #     cv2.waitKey()
    else:
        pass

import paddle.fluid as fluid
class_num = 7
x = fluid.data(name='x', shape=[None, 3, 10], dtype='float32')
label = fluid.data(name='label', shape=[None, 1], dtype='int64')
predict = fluid.layers.fc(input=x, size=class_num, act='softmax')
cost = fluid.layers.cross_entropy(input=predict, label=label)
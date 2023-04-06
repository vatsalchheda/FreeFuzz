import paddle
import paddle.fluid as fluid
import numpy as np
import numpy.random as random

paddle.enable_static()

x = fluid.layers.data(name='x', shape=[2], dtype='float32')
label = fluid.layers.data(name="label", shape=[1], dtype="int64")
y = fluid.layers.fc(input=[x], size=2, act="softmax")
loss = fluid.layers.cross_entropy(input=y, label=label)
loss = paddle.mean(x=loss)
sgd = fluid.optimizer.SGD(learning_rate=0.01)
optimizer = fluid.optimizer.LookaheadOptimizer(sgd,
                                    alpha=0.5,
                                    k=5)
optimizer.minimize(loss)
main_program = fluid.default_main_program()
place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

def train_reader(limit=5):
    for i in range(limit):
        yield random.random([2]).astype('float32'), random.random([1]).astype('int64')

feeder = fluid.DataFeeder(feed_list=[x, label], place=place)
reader = paddle.batch(paddle.reader.shuffle(train_reader, buf_size=50000),batch_size=1)

for batch_data in reader():
    exe.run(fluid.default_main_program(),
    feed=feeder.feed(batch_data))
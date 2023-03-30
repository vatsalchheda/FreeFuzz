import paddle
import paddle.fluid as fluid
import numpy as np

paddle.enable_static()

place = fluid.CPUPlace()
main = fluid.Program()
with fluid.program_guard(main):
    x = paddle.static.data(name='x', shape=[1, 13], dtype='float32')
    y = paddle.static.data(name='y', shape=[1], dtype='float32')
    linear = paddle.nn.Linear(13, 1)
    y_predict = linear(x)
    cost = paddle.nn.functional.square_error_cost(input=y_predict, label=y)
    avg_cost = paddle.mean(cost)

    moment_optimizer = fluid.contrib.optimizer.Momentum(learning_rate=0.001, momentum=0.9)
    moment_optimizer.minimize(avg_cost)

    fetch_list = [avg_cost]
    train_reader = paddle.batch(
        paddle.dataset.uci_housing.train(), batch_size=1)
    feeder = fluid.DataFeeder(place=place, feed_list=[x, y])
    exe = fluid.Executor(place)
    exe.run(paddle.static.default_startup_program())
    for data in train_reader():
        exe.run(main, feed=feeder.feed(data), fetch_list=fetch_list)
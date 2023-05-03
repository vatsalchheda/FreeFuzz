import paddle
paddle.enable_static()
import paddle
import paddle.fluid as fluid
import numpy as np

place = fluid.CPUPlace()
main = fluid.Program()
with fluid.program_guard(main):
    x = fluid.layers.data(name='x', shape=[13], dtype='float32')
    y = fluid.layers.data(name='y', shape=[1], dtype='float32')
    y_predict = fluid.layers.fc(input=x, size=1, act=None)
    cost = fluid.layers.square_error_cost(input=y_predict, label=y)
    avg_cost = fluid.layers.mean(cost)

    ftrl_optimizer = fluid.optimizer.Ftrl(learning_rate=0.1)
    ftrl_optimizer.minimize(avg_cost)

    fetch_list = [avg_cost]
    train_reader = paddle.batch(
        paddle.dataset.uci_housing.train(), batch_size=1)
    feeder = fluid.DataFeeder(place=place, feed_list=[x, y])
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    for data in train_reader():
        exe.run(main, feed=feeder.feed(data), fetch_list=fetch_list)

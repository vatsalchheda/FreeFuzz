import paddle.fluid as fluid
import numpy as np
with fluid.dygraph.guard():
    x = np.random.uniform(-1, 1, [10, 10]).astype("float32")
    linear = fluid.dygraph.Linear(10, 10)
    input = fluid.dygraph.to_variable(x)
    scheduler = fluid.dygraph.LambdaDecay(0.5, lr_lambda=lambda x: 0.95**x)
    adam = fluid.optimizer.Adam(learning_rate = scheduler, parameter_list = linear.parameters())

    for epoch in range(6):
        for batch_id in range(5):
            out = linear(input)
            loss = fluid.layers.reduce_mean(out)
            adam.minimize(loss)
        scheduler.epoch()

        print("epoch:%d, current lr is %f" .format(epoch, adam.current_step_lr()))
        # epoch:0, current lr is 0.5
        # epoch:1, current lr is 0.475
        # epoch:2, current lr is 0.45125
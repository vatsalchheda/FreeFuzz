import paddle.fluid as fluid
import numpy as np
with fluid.dygraph.guard():
    x = np.random.uniform(-1, 1, [10, 10]).astype("float32")
    linear = fluid.dygraph.Linear(10, 10)
    input = fluid.dygraph.to_variable(x)
    scheduler = fluid.dygraph.StepDecay(0.5, step_size=3)
    adam = fluid.optimizer.Adam(learning_rate = scheduler, parameter_list = linear.parameters())

    for epoch in range(9):
        for batch_id in range(5):
            out = linear(input)
            loss = fluid.layers.reduce_mean(out)
            adam.minimize(loss)
        scheduler.epoch()

        print("epoch:{}, current lr is {}" .format(epoch, adam.current_step_lr()))
        # epoch:0, current lr is 0.5
        # epoch:1, current lr is 0.5
        # epoch:2, current lr is 0.5
        # epoch:3, current lr is 0.05
        # epoch:4, current lr is 0.05
        # epoch:5, current lr is 0.05
        # epoch:6, current lr is 0.005
        # epoch:7, current lr is 0.005
        # epoch:8, current lr is 0.005
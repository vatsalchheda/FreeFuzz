import paddle.fluid as fluid
import numpy as np

# example1: LearningRateDecay is not used, return value is all the same
with fluid.dygraph.guard():
    emb = fluid.dygraph.Embedding([10, 10])
    adam = fluid.optimizer.Adam(0.001, parameter_list = emb.parameters())
    lr = adam.current_step_lr()
    print(lr) # 0.001

# example2: PiecewiseDecay is used, return the step learning rate
with fluid.dygraph.guard():
    inp = np.random.uniform(-0.1, 0.1, [10, 10]).astype("float32")
    linear = fluid.dygraph.nn.Linear(10, 10)
    inp = fluid.dygraph.to_variable(inp)
    out = linear(inp)
    loss = fluid.layers.reduce_mean(out)

    bd = [2, 4, 6, 8]
    value = [0.2, 0.4, 0.6, 0.8, 1.0]
    adam = fluid.optimizer.Adam(fluid.dygraph.PiecewiseDecay(bd, value, 0),
                           parameter_list=linear.parameters())

    # first step: learning rate is 0.2
    np.allclose(adam.current_step_lr(), 0.2, rtol=1e-06, atol=0.0) # True

    # learning rate for different steps
    ret = [0.2, 0.2, 0.4, 0.4, 0.6, 0.6, 0.8, 0.8, 1.0, 1.0, 1.0, 1.0]
    for i in range(12):
        adam.minimize(loss)
        lr = adam.current_step_lr()
        np.allclose(lr, ret[i], rtol=1e-06, atol=0.0) # True

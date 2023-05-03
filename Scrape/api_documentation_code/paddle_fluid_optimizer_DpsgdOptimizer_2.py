import paddle.fluid as fluid
import numpy as np

with fluid.dygraph.guard():
    value = np.arange(26).reshape(2, 13).astype("float32")
    a = fluid.dygraph.to_variable(value)
    linear = fluid.Linear(13, 5, dtype="float32")
    # This can be any optimizer supported by dygraph.
    adam = fluid.optimizer.Adam(learning_rate = 0.01,
                                parameter_list = linear.parameters())
    out = linear(a)
    out.backward()
    adam.minimize(out)
    adam.clear_gradients()

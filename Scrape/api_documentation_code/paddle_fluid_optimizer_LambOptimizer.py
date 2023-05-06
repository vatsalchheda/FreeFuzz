import paddle
paddle.enable_static()
import paddle.fluid as fluid

data = fluid.data(name='x', shape=[-1, 5], dtype='float32')
hidden = fluid.layers.fc(input=data, size=10)
cost = fluid.layers.mean(hidden)

def exclude_fn(param):
    return param.name.endswith('.b_0')

optimizer = fluid.optimizer.Lamb(learning_rate=0.002,
                                 exclude_from_weight_decay_fn=exclude_fn)
optimizer.minimize(cost)
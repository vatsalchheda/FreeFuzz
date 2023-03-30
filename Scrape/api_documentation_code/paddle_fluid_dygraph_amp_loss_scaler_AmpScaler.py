import numpy as np
import paddle.fluid as fluid

data = np.random.uniform(-1, 1, [10, 3, 32, 32]).astype('float32')
with fluid.dygraph.guard():
    model = fluid.dygraph.Conv2D(3, 2, 3)
    optimizer = fluid.optimizer.SGDOptimizer(
            learning_rate=0.01, parameter_list=model.parameters())
    scaler = fluid.dygraph.AmpScaler(init_loss_scaling=1024)
    data = fluid.dygraph.to_variable(data)
    with fluid.dygraph.amp_guard():
        conv = model(data)
        loss = fluid.layers.reduce_mean(conv)
        scaled = scaler.scale(loss)
        scaled.backward()
        scaler.minimize(optimizer, scaled)
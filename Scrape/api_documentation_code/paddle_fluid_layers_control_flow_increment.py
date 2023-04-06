import paddle.fluid as fluid
counter = fluid.layers.zeros(shape=[1], dtype='float32') # [0.]
fluid.layers.increment(counter) # [1.]
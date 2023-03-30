import paddle.fluid as fluid
like = fluid.layers.fill_constant(shape=[1,2], value=10, dtype='int64') #like=[[10, 10]]
data = fluid.layers.fill_constant_batch_size_like(
       input=like, shape=[1], value=0, dtype='int64') #like=[[10, 10]] data=[0]
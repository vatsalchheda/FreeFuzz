import paddle.fluid as fluid
# attr shape is a list which doesn't contain  Tensor.
data1 = fluid.layers.fill_constant(shape=[2,1], value=0, dtype='int64') # data1=[[0],[0]]
data2 = fluid.layers.fill_constant(shape=[2,1], value=5, dtype='int64', out=data1)
# data1=[[5], [5]] data2=[[5], [5]]

# attr shape is a list which contains Tensor.
positive_2 = fluid.layers.fill_constant([1], "int32", 2)
data3 = fluid.layers.fill_constant(shape=[1, positive_2], dtype='float32', value=1.5) # data3=[[1.5, 1.5]]

# attr shape is a Tensor.
shape = fluid.layers.fill_constant([2], "int32", 2) # shape=[2,2]
data4 = fluid.layers.fill_constant(shape=shape, dtype='bool', value=True) # data4=[[True,True],[True,True]]

# attr value is a Tensor.
val = fluid.layers.fill_constant([1], "float32", 2.0) # val=[2.0]
data5 = fluid.layers.fill_constant(shape=[2,1], value=val, dtype='float32') #data5=[[2.0],[2.0]]
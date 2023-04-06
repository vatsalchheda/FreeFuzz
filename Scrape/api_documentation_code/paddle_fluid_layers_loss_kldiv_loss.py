import paddle
import paddle.fluid as fluid

x = paddle.rand(shape=[3,4,2,2], dtype='float32')
target = paddle.rand(shape=[3,4,2,2], dtype='float32')

# 'batchmean' reduction, loss shape will be [1]
loss = fluid.layers.kldiv_loss(x=x, target=target, reduction='batchmean')
print(loss.shape) # shape=[1]

# 'mean' reduction, loss shape will be [1]
loss = fluid.layers.kldiv_loss(x=x, target=target, reduction='mean')
print(loss.shape) # shape=[1]

# 'sum' reduction, loss shape will be [1]
loss = fluid.layers.kldiv_loss(x=x, target=target, reduction='sum')
print(loss.shape) # shape=[1]

# 'none' reduction, loss shape is same with X shape
loss = fluid.layers.kldiv_loss(x=x, target=target, reduction='none')
print(loss.shape) # shape=[3, 4, 2, 2]
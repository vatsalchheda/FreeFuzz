import paddle.fluid as fluid

input = fluid.layers.data(name='data', shape=[256], dtype='float32')
label = fluid.layers.data(name='label', shape=[1], dtype='int64')
fc = fluid.layers.fc(input=input, size=100)
out = fluid.layers.sampled_softmax_with_cross_entropy(
          logits=fc, label=label, num_samples=25)
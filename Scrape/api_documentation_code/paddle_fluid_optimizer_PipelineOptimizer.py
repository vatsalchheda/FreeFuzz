import paddle
paddle.enable_static()
import paddle.fluid as fluid
import paddle.fluid.layers as layers

with fluid.device_guard("gpu:0"):
    x = fluid.layers.data(name='x', shape=[1], dtype='int64', lod_level=0)
    y = fluid.layers.data(name='y', shape=[1], dtype='int64', lod_level=0)
    data_loader = fluid.io.DataLoader.from_generator(
        feed_list=[x, y],
        capacity=64,
        use_double_buffer=True,
        iterable=False)

    emb_x = layers.embedding(input=x, param_attr=fluid.ParamAttr(name="embx"), size=[10,2], is_sparse=False)
    emb_y = layers.embedding(input=y, param_attr=fluid.ParamAttr(name="emby",learning_rate=0.9), size=[10,2], is_sparse=False)

with fluid.device_guard("gpu:1"):
    concat = layers.concat([emb_x, emb_y], axis=1)
    fc = layers.fc(input=concat, name="fc", size=1, num_flatten_dims=1, bias_attr=False)
    loss = layers.reduce_mean(fc)
optimizer = fluid.optimizer.SGD(learning_rate=0.5)
optimizer = fluid.optimizer.PipelineOptimizer(optimizer)
optimizer.minimize(loss)

def train_reader():
    for _ in range(4):
        x = np.random.random(size=[1]).astype('int64')
        y = np.random.random(size=[1]).astype('int64')
        yield x, y
data_loader.set_sample_generator(train_reader, batch_size=1)

place = fluid.CUDAPlace(0)
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())
batch_size = 1
data_loader.start()
exe.train_from_dataset(
        fluid.default_main_program())
data_loader.reset()
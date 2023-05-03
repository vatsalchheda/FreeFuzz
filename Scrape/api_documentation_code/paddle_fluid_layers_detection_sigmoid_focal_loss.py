import paddle
paddle.enable_static()
import numpy as np
import paddle.fluid as fluid

num_classes = 10  # exclude background
image_width = 16
image_height = 16
batch_size = 32
max_iter = 20


def gen_train_data():
    x_data = np.random.uniform(0, 255, (batch_size, 3, image_height,
                                        image_width)).astype('float64')
    label_data = np.random.randint(0, num_classes,
                                   (batch_size, 1)).astype('int32')
    return {"x": x_data, "label": label_data}


def get_focal_loss(pred, label, fg_num, num_classes):
    pred = fluid.layers.reshape(pred, [-1, num_classes])
    label = fluid.layers.reshape(label, [-1, 1])
    label.stop_gradient = True
    loss = fluid.layers.sigmoid_focal_loss(
        pred, label, fg_num, gamma=2.0, alpha=0.25)
    loss = fluid.layers.reduce_sum(loss)
    return loss


def build_model(mode='train'):
    x = fluid.data(name="x", shape=[-1, 3, -1, -1], dtype='float64')
    output = fluid.layers.pool2d(input=x, pool_type='avg', global_pooling=True)
    output = fluid.layers.fc(
        input=output,
        size=num_classes,
        # Notice: size is set to be the number of target classes (excluding backgorund)
        # because sigmoid activation will be done in the sigmoid_focal_loss op.
        act=None)
    if mode == 'train':
        label = fluid.data(name="label", shape=[-1, 1], dtype='int32')
        # Obtain the fg_num needed by the sigmoid_focal_loss op:
        # 0 in label represents background, >=1 in label represents foreground,
        # find the elements in label which are greater or equal than 1, then
        # computed the numbers of these elements.
        data = fluid.layers.fill_constant(shape=[1], value=1, dtype='int32')
        fg_label = fluid.layers.greater_equal(label, data)
        fg_label = fluid.layers.cast(fg_label, dtype='int32')
        fg_num = fluid.layers.reduce_sum(fg_label)
        fg_num.stop_gradient = True
        avg_loss = get_focal_loss(output, label, fg_num, num_classes)
        return avg_loss
    else:
        # During evaluating or testing phase,
        # output of the final fc layer should be connected to a sigmoid layer.
        pred = fluid.layers.sigmoid(output)
        return pred


loss = build_model('train')
moment_optimizer = fluid.optimizer.MomentumOptimizer(
    learning_rate=0.001, momentum=0.9)
moment_optimizer.minimize(loss)
place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())
for i in range(max_iter):
    outs = exe.run(feed=gen_train_data(), fetch_list=[loss.name])
    print(outs)
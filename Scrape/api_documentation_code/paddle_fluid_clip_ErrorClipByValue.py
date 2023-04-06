import paddle.fluid as fluid
BATCH_SIZE = 128
CLIP_MAX = 2e-6
CLIP_MIN = -1e-6
prog = fluid.framework.Program()
with fluid.program_guard(main_program=prog):
    image = fluid.layers.data(
        name='x', shape=[784], dtype='float32')
    hidden1 = fluid.layers.fc(input=image, size=128, act='relu')
    hidden2 = fluid.layers.fc(input=hidden1, size=64, act='relu')
    predict = fluid.layers.fc(
        input=hidden2, size=10, act='softmax')
    label = fluid.layers.data(name='y', shape=[1], dtype='int64')
    cost = fluid.layers.cross_entropy(input=predict, label=label)
    avg_cost = fluid.layers.mean(cost)
prog_clip = prog.clone()
prog_clip.block(0).var(hidden1.name)._set_error_clip(
    fluid.clip.ErrorClipByValue(
        max=CLIP_MAX, min=CLIP_MIN)
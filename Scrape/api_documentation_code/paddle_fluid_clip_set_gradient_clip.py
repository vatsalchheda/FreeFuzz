import paddle.fluid as fluid

def network():
    image = fluid.data(name='image', shape=[
                       None, 28], dtype='float32')
    param_attr1 = fluid.ParamAttr("fc1_param")
    fc1 = fluid.layers.fc(image, size=10, param_attr=param_attr1)
    param_attr2 = fluid.ParamAttr("fc2_param")
    fc2 = fluid.layers.fc(fc1, size=10, param_attr=param_attr2)
    loss = fluid.layers.reduce_mean(fc2)
    return loss


# network 1: clip all parameter gradient
with fluid.program_guard(fluid.Program(), fluid.Program()):
    loss = network()
    fluid.clip.set_gradient_clip(
        fluid.clip.GradientClipByGlobalNorm(clip_norm=2.0))
    sgd = fluid.optimizer.SGD(learning_rate=1e-3)
    sgd.minimize(loss)

# network 2: clip parameter gradient by name
with fluid.program_guard(fluid.Program(), fluid.Program()):
    loss = network()
    fluid.clip.set_gradient_clip(
        fluid.clip.GradientClipByValue(min=-1.0, max=1.0),
        param_list=["fc1_param", "fc2_param"])
    sgd = fluid.optimizer.SGD(learning_rate=1e-3)
    sgd.minimize(loss)

# network 3: clip parameter gradient by value
with fluid.program_guard(fluid.Program(), fluid.Program()):
    loss = network()
    param_var1 = fluid.default_main_program().global_block().var("fc1_param")
    param_var2 = fluid.default_main_program().global_block().var("fc2_param")
    fluid.clip.set_gradient_clip(
        fluid.clip.GradientClipByValue(min=-1.0, max=1.0),
        param_list=[param_var1, param_var2])
    sgd = fluid.optimizer.SGD(learning_rate=1e-3)
    sgd.minimize(loss)

# network 4: use 'set_gradient_clip' and 'optimize(grad_clip=clip)' together
with fluid.program_guard(fluid.Program(), fluid.Program()):
    loss = network()
    clip1 = fluid.clip.GradientClipByValue(min=-1.0, max=1.0)
    clip2 = fluid.clip.GradientClipByNorm(clip_norm=1.0)
    # Set the gradient clipping strategy: clip1
    fluid.clip.set_gradient_clip(clip1)
    # Set the gradient clipping strategy: clip2
    sgd = fluid.optimizer.SGD(learning_rate=1e-3, grad_clip=clip2)
    sgd.minimize(loss)
    # 'set_gradient_clip' will not take effect when setting has a conflict,
    # and the gradient clipping strategy will be 'clip2'
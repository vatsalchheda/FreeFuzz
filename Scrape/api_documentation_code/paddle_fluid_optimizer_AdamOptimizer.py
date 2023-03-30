# Adam with beta1/beta2 as Variable
import paddle
import paddle.fluid as fluid
import paddle.fluid.layers.learning_rate_scheduler as lr_scheduler

place = fluid.CPUPlace()
main = fluid.Program()
with fluid.program_guard(main):
    x = fluid.data(name='x', shape=[None, 13], dtype='float32')
    y = fluid.data(name='y', shape=[None, 1], dtype='float32')
    y_predict = fluid.layers.fc(input=x, size=1, act=None)
    cost = fluid.layers.square_error_cost(input=y_predict, label=y)
    avg_cost = fluid.layers.mean(cost)

    # define beta decay variable
    def get_decayed_betas(beta1_init, beta2_init, decay_steps, decay_rate, epsilon_init):
        global_step = lr_scheduler._decay_step_counter()

        beta1 = fluid.layers.create_global_var(
            shape=[1],
            value=float(beta1_init),
            dtype='float32',
            # set persistable for save checkpoints and resume
            persistable=True,
            name="beta1")
        beta2 = fluid.layers.create_global_var(
            shape=[1],
            value=float(beta2_init),
            dtype='float32',
            # set persistable for save checkpoints and resume
            persistable=True,
            name="beta2")
        epsilon = fluid.layers.create_global_var(
            shape=[1],
            value=float(epsilon_init),
            dtype='float32',
            # set persistable for save checkpoints and resume
            persistable=True,
            name="epsilon")

        div_res = global_step / decay_steps
        decayed_beta1 = beta1_init * (decay_rate**div_res)
        decayed_beta2 = beta2_init * (decay_rate**div_res)
        fluid.layers.assign(decayed_beta1, beta1)
        fluid.layers.assign(decayed_beta2, beta2)

        return beta1, beta2, epsilon

    beta1, beta2, epsilon = get_decayed_betas(0.9, 0.99, 1e5, 0.9, 1e-8)
    adam_optimizer = fluid.optimizer.AdamOptimizer(
                                        learning_rate=0.01,
                                        beta1=beta1,
                                        beta2=beta2,
                                        epsilon=epsilon)
    adam_optimizer.minimize(avg_cost)

    fetch_list = [avg_cost]
    train_reader = paddle.batch(
        paddle.dataset.uci_housing.train(), batch_size=1)
    feeder = fluid.DataFeeder(place=place, feed_list=[x, y])
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    for data in train_reader():
        exe.run(main, feed=feeder.feed(data), fetch_list=fetch_list)
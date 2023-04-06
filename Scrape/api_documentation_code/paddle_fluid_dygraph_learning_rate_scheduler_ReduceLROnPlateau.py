import paddle.fluid as fluid
import numpy as np

with fluid.dygraph.guard():
    x = np.random.uniform(-1, 1, [10, 10]).astype("float32")
    linear = fluid.dygraph.Linear(10, 10)
    input = fluid.dygraph.to_variable(x)

    reduce_lr = fluid.dygraph.ReduceLROnPlateau(
                            learning_rate = 1.0,
                            decay_rate = 0.5,
                            patience = 5,
                            verbose = True,
                            cooldown = 3)
    adam = fluid.optimizer.Adam(
        learning_rate = reduce_lr,
        parameter_list = linear.parameters())

    for epoch in range(10):
        total_loss = 0
        for bath_id in range(5):
            out = linear(input)
            loss = fluid.layers.reduce_mean(out)
            total_loss += loss
            adam.minimize(loss)

        avg_loss = total_loss/5

        # adjust learning rate according to avg_loss
        reduce_lr.step(avg_loss)
        lr = adam.current_step_lr()
        print("current avg_loss is %s, current lr is %s" % (avg_loss.numpy()[0], lr))
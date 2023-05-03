import paddle.fluid as fluid

with fluid.dygraph.guard():
    linear = fluid.dygraph.nn.Linear(10, 10)

    adam = fluid.optimizer.Adam(0.1, parameter_list=linear.parameters())

    # set learning rate manually by python float value
    lr_list = [0.2, 0.3, 0.4, 0.5, 0.6]
    for i in range(5):
        adam.set_lr(lr_list[i])
        lr = adam.current_step_lr()
        print("current lr is {}".format(lr))
    # Print:
    #    current lr is 0.2
    #    current lr is 0.3
    #    current lr is 0.4
    #    current lr is 0.5
    #    current lr is 0.6


    # set learning rate manually by framework Variable
    lr_var = fluid.layers.create_global_var(
        shape=[1], value=0.7, dtype='float32')
    adam.set_lr(lr_var)
    lr = adam.current_step_lr()
    print("current lr is {}".format(lr))
    # Print:
    #    current lr is 0.7

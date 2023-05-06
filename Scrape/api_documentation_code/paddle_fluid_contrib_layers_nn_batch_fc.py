import paddle
paddle.enable_static()
import paddle.fluid as fluid
input = fluid.data(name="input", shape=[16, 2, 3], dtype="float32")
out = fluid.contrib.layers.batch_fc(input=input,
                                    param_size=[16, 3, 10],
                                    param_attr=
                                      fluid.ParamAttr(learning_rate=1.0,
                                                    name="w_0",
                                                    initializer=
                                                    fluid.initializer.Xavier(uniform=False)),
                                    bias_size=[16, 10],
                                    bias_attr=
                                      fluid.ParamAttr(learning_rate=1.0,
                                                    name="b_0",
                                                    initializer=
                                                    fluid.initializer.Xavier(uniform=False)),
                                        act="relu")
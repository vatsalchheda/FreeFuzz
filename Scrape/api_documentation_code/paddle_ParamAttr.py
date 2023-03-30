import paddle

weight_attr = paddle.ParamAttr(name="weight",
                               learning_rate=0.5,
                               regularizer=paddle.regularizer.L2Decay(1.0),
                               trainable=True)
print(weight_attr.name) # "weight"
paddle.nn.Linear(3, 4, weight_attr=weight_attr)
import paddle

x = paddle.uniform([10, 10], min=-1.0, max=1.0, dtype='float32')
linear = paddle.nn.Linear(in_features=10, out_features=10,
                          weight_attr=paddle.ParamAttr(need_clip=True),
                          bias_attr=paddle.ParamAttr(need_clip=False))
out = linear(x)
loss = paddle.mean(out)
loss.backward()

clip = paddle.nn.ClipGradByNorm(clip_norm=1.0)
sdg = paddle.optimizer.SGD(learning_rate=0.1, parameters=linear.parameters(), grad_clip=clip)
sdg.step()
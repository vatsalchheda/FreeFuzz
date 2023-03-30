import paddle

model = paddle.nn.Conv2D(3, 2, 3, bias_attr=True)
optimizer = paddle.optimizer.SGD(learning_rate=0.01, parameters=model.parameters())
scaler = paddle.amp.GradScaler(init_loss_scaling=1024)
data = paddle.rand([10, 3, 32, 32])

with paddle.amp.auto_cast():
    conv = model(data)
    loss = paddle.mean(conv)

scaled = scaler.scale(loss)  # scale the loss
scaled.backward()            # do backward
scaler.minimize(optimizer, scaled)  # update parameters
optimizer.clear_grad()
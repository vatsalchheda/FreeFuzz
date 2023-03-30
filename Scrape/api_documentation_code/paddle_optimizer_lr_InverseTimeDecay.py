import paddle
import numpy as np

# train on default dynamic graph mode
linear = paddle.nn.Linear(10, 10)
scheduler = paddle.optimizer.lr.InverseTimeDecay(learning_rate=0.5, gamma=0.1, verbose=True)
sgd = paddle.optimizer.SGD(learning_rate=scheduler, parameters=linear.parameters())
for epoch in range(20):
    for batch_id in range(5):
        x = paddle.uniform([10, 10])
        out = linear(x)
        loss = paddle.mean(out)
        loss.backward()
        sgd.step()
        sgd.clear_gradients()
        scheduler.step()    # If you update learning rate each step
  # scheduler.step()        # If you update learning rate each epoch

# train on static graph mode
paddle.enable_static()
main_prog = paddle.static.Program()
start_prog = paddle.static.Program()
with paddle.static.program_guard(main_prog, start_prog):
    x = paddle.static.data(name='x', shape=[None, 4, 5])
    y = paddle.static.data(name='y', shape=[None, 4, 5])
    z = paddle.static.nn.fc(x, 100)
    loss = paddle.mean(z)
    scheduler = paddle.optimizer.lr.InverseTimeDecay(learning_rate=0.5, gamma=0.1, verbose=True)
    sgd = paddle.optimizer.SGD(learning_rate=scheduler)
    sgd.minimize(loss)

exe = paddle.static.Executor()
exe.run(start_prog)
for epoch in range(20):
    for batch_id in range(5):
        out = exe.run(
            main_prog,
            feed={
                'x': np.random.randn(3, 4, 5).astype('float32'),
                'y': np.random.randn(3, 4, 5).astype('float32')
            },
            fetch_list=loss.name)
        scheduler.step()    # If you update learning rate each step
  # scheduler.step()        # If you update learning rate each epoch
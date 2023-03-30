import numpy as np
import paddle

paddle.enable_static()
paddle.incubate.autograd.enable_prim()

startup_program = paddle.static.Program()
main_program = paddle.static.Program()

with paddle.static.program_guard(main_program, startup_program):
    x = paddle.static.data('x', shape=[1], dtype='float32')
    y = x * x
    y_grad = paddle.incubate.autograd.forward_grad(y, x)
    paddle.incubate.autograd.prim2orig()

exe = paddle.static.Executor()
exe.run(startup_program)
y_grad = exe.run(main_program, feed={'x': np.array([2.]).astype('float32')}, fetch_list=[y_grad])
print(y_grad)
# [array([4.], dtype=float32)]

paddle.incubate.autograd.disable_prim()
paddle.disable_static()
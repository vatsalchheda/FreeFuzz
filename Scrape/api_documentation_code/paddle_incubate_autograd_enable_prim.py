import paddle
from paddle.incubate.autograd import enable_prim, prim_enabled

paddle.enable_static()
enable_prim()

print(prim_enabled()) # True
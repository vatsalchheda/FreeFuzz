# required: distributed
import numpy
import paddle
import paddle.distributed as dist
from paddle.autograd import PyLayer
from paddle.distributed.fleet.utils.hybrid_parallel_util import fused_allreduce_gradients

class cus_tanh(PyLayer):
    @staticmethod
    def forward(ctx, x):
        y = paddle.tanh(x)
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, dy):
        y, = ctx.saved_tensor()
        grad = dy * (1 - paddle.square(y))
        return grad

class SimpleNet(paddle.nn.Layer):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.linear = paddle.nn.Linear(2, 2)

    def forward(self, inputs):
        inputs = cus_tanh.apply(inputs)
        return self.linear(inputs)

if __name__ == '__main__':
    dist.init_parallel_env()

    model = SimpleNet()
    model = paddle.DataParallel(model)
    opt = paddle.optimizer.SGD(learning_rate=0.01, parameters=model.parameters())

    for step in range(10):
        x_data = numpy.random.randn(2, 2).astype(numpy.float32)
        x = paddle.to_tensor(x_data)
        x.stop_gradient = False

        # step 1 : skip gradient synchronization by 'no_sync'
        with model.no_sync():
            y_pred = model(x)
            loss = y_pred.mean()
            loss.backward()

        # step 2 : fuse + allreduce manually before optimization
        fused_allreduce_gradients(list(model.parameters()), None)

        opt.step()
        opt.clear_grad()
import paddle
import paddle.vision.transforms as T
from paddle.vision.datasets import MNIST
from paddle.static import InputSpec

inputs = [InputSpec([-1, 1, 28, 28], 'float32', 'image')]
labels = [InputSpec([None, 1], 'int64', 'label')]

transform = T.Compose([
    T.Transpose(),
    T.Normalize([127.5], [127.5])
])
train_dataset = MNIST(mode='train', transform=transform)

lenet = paddle.vision.models.LeNet()
model = paddle.Model(lenet,
    inputs, labels)

optim = paddle.optimizer.Adam(0.001, parameters=lenet.parameters())
model.prepare(optimizer=optim,
            loss=paddle.nn.CrossEntropyLoss(),
            metrics=paddle.metric.Accuracy())

callback = paddle.callbacks.ProgBarLogger(log_freq=10)
model.fit(train_dataset, batch_size=64, callbacks=callback)
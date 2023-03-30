import paddle
from paddle import Model
from paddle.static import InputSpec
from paddle.vision.models import LeNet
from paddle.vision.datasets import MNIST
from paddle.metric import Accuracy
from paddle.nn import CrossEntropyLoss
import paddle.vision.transforms as T

device = paddle.set_device('cpu')
sample_num = 200
save_dir = './best_model_checkpoint'
transform = T.Compose(
    [T.Transpose(), T.Normalize([127.5], [127.5])])
train_dataset = MNIST(mode='train', transform=transform)
val_dataset = MNIST(mode='test', transform=transform)
net = LeNet()
optim = paddle.optimizer.Adam(
    learning_rate=0.001, parameters=net.parameters())

inputs = [InputSpec([None, 1, 28, 28], 'float32', 'x')]
labels = [InputSpec([None, 1], 'int64', 'label')]

model = Model(net, inputs=inputs, labels=labels)
model.prepare(
    optim,
    loss=CrossEntropyLoss(reduction="sum"),
    metrics=[Accuracy()])
callbacks = paddle.callbacks.EarlyStopping(
    'loss',
    mode='min',
    patience=1,
    verbose=1,
    min_delta=0,
    baseline=None,
    save_best_model=True)
model.fit(train_dataset,
          val_dataset,
          batch_size=64,
          log_freq=200,
          save_freq=10,
          save_dir=save_dir,
          epochs=20,
          callbacks=[callbacks])
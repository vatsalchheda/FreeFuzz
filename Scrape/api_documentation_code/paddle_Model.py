# required: gpu
import paddle
import paddle.nn as nn
import paddle.vision.transforms as T

def run_example_code():
    device = paddle.set_device('gpu')

    net = nn.Sequential(nn.Flatten(1), nn.Linear(784, 200), nn.Tanh(),
                        nn.Linear(200, 10))

    model = paddle.Model(net)
    optim = paddle.optimizer.SGD(learning_rate=1e-3, parameters=model.parameters())

    amp_configs = {
        "level": "O1",
        "custom_white_list": {'conv2d'},
        "use_dynamic_loss_scaling": True
    }
    model.prepare(optim,
        paddle.nn.CrossEntropyLoss(),
        paddle.metric.Accuracy(),
        amp_configs=amp_configs)

    transform = T.Compose([T.Transpose(), T.Normalize([127.5], [127.5])])
    data = paddle.vision.datasets.MNIST(mode='train', transform=transform)
    model.fit(data, epochs=2, batch_size=32, verbose=1)

# mixed precision training is only supported on GPU now.
if paddle.is_compiled_with_cuda():
    run_example_code()
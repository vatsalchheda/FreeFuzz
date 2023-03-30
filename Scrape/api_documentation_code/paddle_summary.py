import paddle
import paddle.nn as nn

class LeNet(nn.Layer):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2D(
                1, 6, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2D(2, 2),
            nn.Conv2D(
                6, 16, 5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2D(2, 2))

        if num_classes > 0:
            self.fc = nn.Sequential(
                nn.Linear(400, 120),
                nn.Linear(120, 84),
                nn.Linear(
                    84, 10))

    def forward(self, inputs):
        x = self.features(inputs)

        if self.num_classes > 0:
            x = paddle.flatten(x, 1)
            x = self.fc(x)
        return x

lenet = LeNet()

params_info = paddle.summary(lenet, (1, 1, 28, 28))
print(params_info)

# multi input demo
class LeNetMultiInput(LeNet):

    def forward(self, inputs, y):
        x = self.features(inputs)

        if self.num_classes > 0:
            x = paddle.flatten(x, 1)
            x = self.fc(x + y)
        return x

lenet_multi_input = LeNetMultiInput()

params_info = paddle.summary(lenet_multi_input, [(1, 1, 28, 28), (1, 400)],
                            dtypes=['float32', 'float32'])
print(params_info)

# list input demo
class LeNetListInput(LeNet):

    def forward(self, inputs):
        x = self.features(inputs[0])

        if self.num_classes > 0:
            x = paddle.flatten(x, 1)
            x = self.fc(x + inputs[1])
        return x

lenet_list_input = LeNetListInput()
input_data = [paddle.rand([1, 1, 28, 28]), paddle.rand([1, 400])]
params_info = paddle.summary(lenet_list_input, input=input_data)
print(params_info)

# dict input demo
class LeNetDictInput(LeNet):

    def forward(self, inputs):
        x = self.features(inputs['x1'])

        if self.num_classes > 0:
            x = paddle.flatten(x, 1)
            x = self.fc(x + inputs['x2'])
        return x

lenet_dict_input = LeNetDictInput()
input_data = {'x1': paddle.rand([1, 1, 28, 28]),
              'x2': paddle.rand([1, 400])}
params_info = paddle.summary(lenet_dict_input, input=input_data)
print(params_info)
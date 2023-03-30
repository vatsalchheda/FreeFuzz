import paddle
import paddle.nn.functional as F

x = paddle.randn((3,224,224,2))
label = paddle.randint(high=2, shape=(3,224,224,1))
predictions = F.softmax(x)
loss = F.dice_loss(input=predictions, label=label)
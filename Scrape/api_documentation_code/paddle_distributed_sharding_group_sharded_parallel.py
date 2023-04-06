# required: distributed
import paddle
from paddle.fluid.dygraph.nn import Linear
from paddle.distributed import fleet
from paddle.distributed.sharding import group_sharded_parallel

fleet.init(is_collective=True)
group = paddle.distributed.new_group([0, 1])
model = Linear(1000, 1000)

clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=1.0)
optimizer = paddle.optimizer.AdamW(learning_rate=0.001, parameters=model.parameters(), weight_decay=0.00001, grad_clip=clip)

# wrap sharding model, optimizer and scaler
model, optimizer, scaler = group_sharded_parallel(model, optimizer, "p_g", scaler=scaler)

img, label = data
label.stop_gradient = True
img.stop_gradient = True

out = model(img)
loss = paddle.nn.functional.cross_entropy(input=out, label=label)

loss.backward()
optimizer.step()
optimizer.clear_grad()
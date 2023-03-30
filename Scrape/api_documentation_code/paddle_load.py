# example 1: dynamic graph
import paddle
emb = paddle.nn.Embedding(10, 10)
layer_state_dict = emb.state_dict()

# save state_dict of emb
paddle.save(layer_state_dict, "emb.pdparams")

scheduler = paddle.optimizer.lr.NoamDecay(
    d_model=0.01, warmup_steps=100, verbose=True)
adam = paddle.optimizer.Adam(
    learning_rate=scheduler,
    parameters=emb.parameters())
opt_state_dict = adam.state_dict()

# save state_dict of optimizer
paddle.save(opt_state_dict, "adam.pdopt")
# save weight of emb
paddle.save(emb.weight, "emb.weight.pdtensor")

# load state_dict of emb
load_layer_state_dict = paddle.load("emb.pdparams")
# load state_dict of optimizer
load_opt_state_dict = paddle.load("adam.pdopt")
# load weight of emb
load_weight = paddle.load("emb.weight.pdtensor")


# example 2: Load multiple state_dict at the same time
from paddle import nn
from paddle.optimizer import Adam

layer = paddle.nn.Linear(3, 4)
adam = Adam(learning_rate=0.001, parameters=layer.parameters())
obj = {'model': layer.state_dict(), 'opt': adam.state_dict(), 'epoch': 100}
path = 'example/model.pdparams'
paddle.save(obj, path)
obj_load = paddle.load(path)


# example 3: static graph
import paddle
import paddle.static as static

paddle.enable_static()

# create network
x = paddle.static.data(name="x", shape=[None, 224], dtype='float32')
z = paddle.static.nn.fc(x, 10)

place = paddle.CPUPlace()
exe = paddle.static.Executor(place)
exe.run(paddle.static.default_startup_program())
prog = paddle.static.default_main_program()
for var in prog.list_vars():
    if list(var.shape) == [224, 10]:
        tensor = var.get_value()
        break

# save/load tensor
path_tensor = 'temp/tensor.pdtensor'
paddle.save(tensor, path_tensor)
load_tensor = paddle.load(path_tensor)

# save/load state_dict
path_state_dict = 'temp/model.pdparams'
paddle.save(prog.state_dict("param"), path_tensor)
load_state_dict = paddle.load(path_tensor)


# example 4: load program
import paddle

paddle.enable_static()

data = paddle.static.data(
    name='x_static_save', shape=(None, 224), dtype='float32')
y_static = z = paddle.static.nn.fc(data, 10)
main_program = paddle.static.default_main_program()
path = "example/main_program.pdmodel"
paddle.save(main_program, path)
load_main = paddle.load(path)
print(load_main)


# example 5: save object to memory
from io import BytesIO
import paddle
from paddle.nn import Linear
paddle.disable_static()

linear = Linear(5, 10)
state_dict = linear.state_dict()
byio = BytesIO()
paddle.save(state_dict, byio)
tensor = paddle.randn([2, 3], dtype='float32')
paddle.save(tensor, byio)
byio.seek(0)
# load state_dict
dict_load = paddle.load(byio)
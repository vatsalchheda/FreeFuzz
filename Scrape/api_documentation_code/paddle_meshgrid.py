import paddle

x = paddle.randint(low=0, high=100, shape=[100])
y = paddle.randint(low=0, high=100, shape=[200])

grid_x, grid_y = paddle.meshgrid(x, y)

print(grid_x.shape)
print(grid_y.shape)

#the shape of res_1 is (100, 200)
#the shape of res_2 is (100, 200)
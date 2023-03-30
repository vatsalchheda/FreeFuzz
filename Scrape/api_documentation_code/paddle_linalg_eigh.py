import paddle

x = paddle.to_tensor([[1, -2j], [2j, 5]])
out_value, out_vector = paddle.linalg.eigh(x, UPLO='L')
print(out_value)
#[0.17157288, 5.82842712]
print(out_vector)
#[(-0.9238795325112867+0j), (-0.3826834323650898+0j)],
#[ 0.3826834323650898j    , -0.9238795325112867j    ]]
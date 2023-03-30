import paddle

paddle.device.set_device("cpu")

x = paddle.to_tensor([[1.6707249, 7.2249975, 6.5045543],
                   [9.956216,  8.749598,  6.066444 ],
                   [4.4251957, 1.7983172, 0.370647 ]])
w, v = paddle.linalg.eig(x)
print(v)
# Tensor(shape=[3, 3], dtype=complex128, place=CPUPlace, stop_gradient=False,
#       [[(-0.5061363550800655+0j) , (-0.7971760990842826+0j) ,
#         (0.18518077798279986+0j)],
#        [(-0.8308237755993192+0j) ,  (0.3463813401919749+0j) ,
#         (-0.6837005269141947+0j) ],
#        [(-0.23142567697893396+0j),  (0.4944999840400175+0j) ,
#         (0.7058765252952796+0j) ]])

print(w)
# Tensor(shape=[3], dtype=complex128, place=CPUPlace, stop_gradient=False,
#       [ (16.50471283351188+0j)  , (-5.5034820550763515+0j) ,
#         (-0.21026087843552282+0j)])
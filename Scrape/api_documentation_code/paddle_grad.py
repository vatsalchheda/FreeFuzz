import paddle

def test_dygraph_grad(grad_outputs=None):
    x = paddle.to_tensor(2.0)
    x.stop_gradient = False

    y1 = x * x
    y2 = x * 3

    # If grad_outputs=None, dy1 = [1], dy2 = [1].
    # If grad_outputs=[g1, g2], then:
    #    - dy1 = [1] if g1 is None else g1
    #    - dy2 = [1] if g2 is None else g2

    # Since y1 = x * x, dx = 2 * x * dy1.
    # Since y2 = x * 3, dx = 3 * dy2.
    # Therefore, the final result would be:
    # dx = 2 * x * dy1 + 3 * dy2 = 4 * dy1 + 3 * dy2.

    dx = paddle.grad(
        outputs=[y1, y2],
        inputs=[x],
        grad_outputs=grad_outputs)[0]

    return dx.numpy()

grad_value = paddle.to_tensor(4.0)
# dy1 = [1], dy2 = [1]
print(test_dygraph_grad(None)) # [7.]

# dy1 = [1], dy2 = [4]
print(test_dygraph_grad([None, grad_value])) # [16.]

# dy1 = [4], dy2 = [1]
print(test_dygraph_grad([grad_value, None])) # [19.]

# dy1 = [3], dy2 = [4]
grad_y1 = paddle.to_tensor(3.0)
print(test_dygraph_grad([grad_y1, grad_value])) # [24.]
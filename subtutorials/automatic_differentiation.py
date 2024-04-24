# https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html

# When training neural networks, the most frequently used algorithm is back
# propagation. In this algorithm, parameters (model weights) are adjusted
# according to the gradient of the loss function with respect to the given
# parameter.
#
# To compute those gradients, PyTorch has a built-in differentiation engine
# called torch.autograd. It supports automatic computation of gradient for any
# computational graph.

# Consider the simplest one-layer neural network, with input x, parameters w
# and b, and some loss function. It can be defined in PyTorch in the following
# manner:

import torch

x = torch.ones(5)  # input tensor
# tensor([1., 1., 1., 1., 1.])

y = torch.zeros(3)  # expected output
# tensor([0., 0., 0.])

w = torch.randn(5, 3, requires_grad=True)
# tensor([[ 0.5928, -1.7439,  0.2688],
#         [ 0.3934, -0.4670, -0.9636],
#         [ 2.7544, -0.3819, -0.7877],
#         [ 0.9970, -0.1971,  1.3086],
#         [-1.0546,  0.0124, -0.3287]], requires_grad=True)

b = torch.randn(3, requires_grad=True)
# tensor([ 0.0560, -0.4197,  2.9129], requires_grad=True)

z = torch.matmul(x, w)+b
# tensor([ 3.7390, -3.1971,  2.4103], grad_fn=<AddBackward0>)

loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
# tensor(2.0996, grad_fn=<BinaryCrossEntropyWithLogitsBackward0>)

print(f"Gradient function for z = {z.grad_fn}")
print(f"Gradient function for loss = {loss.grad_fn}")

loss.backward()
print(w.grad)
print(b.grad)
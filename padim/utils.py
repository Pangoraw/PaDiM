"""
Utils module

The code from this file comes from:
    * https://github.com/taikiinoue45/PaDiM
"""
import torch
import torch.nn.functional as F


def embeddings_concat(x0, x1):
    b0, c0, h0, w0 = x0.size()
    _, c1, h1, w1 = x1.size()
    s = h0 // h1
    x0 = F.unfold(x0, kernel_size=(s, s), dilation=(1, 1), stride=(s, s))
    x0 = x0.view(b0, c0, -1, h1, w1)
    z = torch.zeros(b0, c0 + c1, x0.size(2), h1, w1)
    for i in range(x0.size(2)):
        z[:, :, i, :, :] = torch.cat((x0[:, :, i, :, :], x1), 1)
    z = z.view(b0, -1, h1 * w1)
    z = F.fold(z, kernel_size=(s, s), output_size=(h0, w0), stride=(s, s))
    return z

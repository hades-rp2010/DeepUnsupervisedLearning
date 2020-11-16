import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedLayer(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(MaskedLayer, self).__init__(in_features, out_features, bias)
        self.register_buffer("mask", torch.ones(out_features, in_features))

    def set_mask(self, mask):
        self.mask.data.copy_(torch.as_tensor(mask.T))

    def forward(self, x):
        return F.linear(x, self.mask * self.weight, self.bias)


class MADE(nn.Module):
    def __init__(self, n_inputs, hidden_sizes, n_outputs, n_masks=1):
        super(MADE, self).__init__()
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.hidden = hidden_sizes
        self.n_masks = n_masks
        self.seed = 0
        self.m = {}
        
        layers = [self.n_inputs] + list(self.hidden) + [self.n_outputs]
        ls = []
        for i in range(len(layers) - 1):
            ls.append(MaskedLayer(layers[i], layers[i+1]))
            ls.append(nn.ReLU())

        # Get rid of the final relu layer
        ls = ls[:-1]

        self.network = nn.Sequential(*ls)
        self.update_masks()

    def update_masks(self):
        rnd = np.random.RandomState(self.seed)
        self.seed = (self.seed + 1) % self.n_masks

        self.m[-1] = rnd.permutation(self.n_inputs)
        
        for l in range(len(self.hidden)):
            self.m[l] = rnd.randint(self.m[l-1].min(), self.n_inputs-1, size=self.hidden[l])
        
        masks = [self.m[l-1][:, None] <= self.m[l][None, :] for l in range(len(self.hidden))]
        masks.append(self.m[len(self.hidden) - 1][:, None] < self.m[-1][None, :])
        

        # The case where n_outputs = k * n_inputs for k > 1
        if self.n_outputs > self.n_inputs:
            k = int(self.n_outputs / self.n_inputs)
            # replicate the same mask
            masks[-1] = np.concatenate([masks[-1]] * k, axis=1)

        layers = [l for l in self.network.modules() if isinstance(l, MaskedLayer)]

        for l, m in zip(layers, masks):
            l.set_mask(m)

    def forward(self, x):
        return self.network(x)


if __name__ == '__main__':
    from torch.autograd import Variable
    
    # run a quick and dirty test for the autoregressive property
    D = 10
    rnd = np.random.RandomState(14)
    x = (rnd.rand(1, D) > 0.5).astype(np.float32)
    
    configs = [
        (D, [], D),                 # test various hidden sizes
        (D, [200], D),
        (D, [200, 220], D),
        (D, [200, 220, 230], D),
        (D, [200, 220], D),          # natural ordering test
        (D, [200, 220], 2*D),       # test n_outputs > n_inputs
        (D, [200, 220], 3*D),       # test n_outputs > n_inputs
    ]
    
    for nin, hiddens, nout in configs:
        
        print(f"checking n_inputs {nin}, hidden {hiddens}, n_outputs {nout}")
        model = MADE(nin, hiddens, nout)
        
        # run backpropagation for each dimension to compute what other
        # dimensions it depends on.
        res = []
        for k in range(nout):
            xtr = Variable(torch.from_numpy(x), requires_grad=True)
            xtrhat = model(xtr)
            loss = xtrhat[0,k]
            loss.backward()
            
            depends = (xtr.grad[0].numpy() != 0).astype(np.uint8)
            depends_ix = list(np.where(depends)[0])
            isok = k % nin not in depends_ix
            
            res.append((len(depends_ix), k, depends_ix, isok))
        
        # pretty print the dependencies
        res.sort()
        for nl, k, ix, isok in res:
            print("output %2d depends on inputs: %30s : %s" % (k, ix, "OK" if isok else "NOTOK"))
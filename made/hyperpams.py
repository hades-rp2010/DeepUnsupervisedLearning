import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt

from made import MADE


def run_one_epoch(split, verbose=True, upto=None):
    model.train() if split == "train" else model.eval()
    x = x_train if split == "train" else x_test
    n_samples = 1
    batch_size = 100
    
    n_steps = x.size(0) // batch_size if upto is None else np.min([x.size(0) // batch_size, upto])
    losses = []

    for i in range(n_steps):
        x_batch = Variable(x[i * batch_size:(i+1) * batch_size])
        x_hat = torch.zeros_like(x_batch)

        for s in range(n_samples):
            if i % resample_every == 0 or split == "test":
                model.update_masks
            
            # forward pass
            x_hat = model(x_batch)
        x_batch / n_samples

        # evaluate the loss and update
        loss = nn.BCEWithLogitsLoss(size_average=False)(x_hat, x_batch) / batch_size
        losses.append(loss.data.item())

        if split == "train":
            opt.zero_grad()
            loss.backward()
            opt.step()

    if verbose:
        print(f"{split} Average epoch loss: {np.mean(losses)}")
    return losses


if __name__ == "__main__":
    # load the dataset from some path
    mnist = np.load("binarized_mnist.npz")
    x_train, x_test = mnist["train_data"], mnist["valid_data"]
    x_train = torch.as_tensor(x_train).cuda()
    x_test = torch.as_tensor(x_test).cuda()

    hidden_list = [500]
    resample_every = 20
    masks = np.arange(1, 10)
    test_losses = []
    for i in range(len(masks)):
        model = MADE(x_train.size(1), hidden_list, x_train.size(1), n_masks=masks[i])
        print(f"number of model parameters: {np.sum([np.prod(p.size()) for p in model.parameters()])}")
        model.cuda()

        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=50, gamma=0.1)

        # The training
        for epoch in range(100):
            print(f"Epoch {epoch}")
            scheduler.step()

            # get an estimate of the test loss
            # run_one_epoch("test", upto=5)
            run_one_epoch("train")

        print("Final test eval:")
        loss = run_one_epoch("test")
        test_losses.append(loss)

    plt.plot(test_losses)
    plt.show()
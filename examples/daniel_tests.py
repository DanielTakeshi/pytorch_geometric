"""Doing some brief tests."""
import os.path as osp
import time
import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP, PointConv, fps, global_max_pool, radius


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(epoch):
    model.train()

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        loss = F.nll_loss(model(data), data.y)
        loss.backward()
        optimizer.step()


def test(loader):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data).max(1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)


if __name__ == '__main__':
    # Load ModelNet10 if it's not there already. Handle data loaders, etc.
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data/ModelNet10')
    pre_transform, transform = T.NormalizeScale(), T.SamplePoints(1024)
    train_dataset = ModelNet(path, '10', True, transform, pre_transform)
    test_dataset = ModelNet(path, '10', False, transform, pre_transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                              num_workers=6)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False,
                             num_workers=6)

    # Create the model and optimizer.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print(f'The classification model:\n{model}')
    print(f'Parameters: {count_parameters(model)}.\n')
    start = time.time()
    for epoch in range(1, 201):
        train(epoch)
        test_acc = test(test_loader)
        print(f'Epoch: {epoch:03d}, Test: {test_acc:.4f}')
    elapsed = time.time() - start
    print(f'Elapsed time: {elapsed:0.1f}s')

import time
import pickle
import os.path as osp
from collections import defaultdict
import numpy as np

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP, PointConv, fps, global_max_pool, radius


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointConv(nn, add_self_loops=False)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=64)
        #print(pos.shape, pos[idx].shape, row.shape, col.shape, pos[idx].shape[0]*64)
        #import pdb; pdb.set_trace()
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super().__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


class Net(torch.nn.Module):

    def __init__(self, out_channels):
        super().__init__()

        # Input channels account for both `pos` and node features. NOTE(daniel):
        # it is important that the MLP for `sa1_module` has '3' here because
        # that's the dimension of the PC data (data.pos is 3D, data.x is None).
        # NOTE(daniel): can crank up the radius values (0.2 and 0.4 by default) to
        # some crazily high # and it doesn't matter (code only takes 64 points,
        # though not clear if the 64 _nearest_ such points...).
        self.sa1_module = SAModule(0.5, 0.2, MLP([3, 64, 64, 128]))
        self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))

        self.mlp = MLP([1024, 512, 256, out_channels], dropout=0.5, batch_norm=False)

    def forward(self, x, pos, batch):
        """Daniel: changing this to explicitly have (x, pos, batch)."""
        sa0_out = (x, pos, batch)
        #print(f'sa0_out: (sa0_out[0]=None) {sa0_out[1].shape} {sa0_out[2].shape}')
        sa1_out = self.sa1_module(*sa0_out)
        #print(f'sa1_out: {sa1_out[0].shape} {sa1_out[1].shape} {sa1_out[2].shape}')
        sa2_out = self.sa2_module(*sa1_out)
        #print(f'sa2_out: {sa2_out[0].shape} {sa2_out[1].shape} {sa2_out[2].shape}')
        sa3_out = self.sa3_module(*sa2_out)
        #print(f'sa3_out: {sa3_out[0].shape} {sa3_out[1].shape} {sa3_out[2].shape}')
        x, pos, batch = sa3_out
        #import pdb; pdb.set_trace()
        return self.mlp(x).log_softmax(dim=-1)


def train():
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.pos, data.batch)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        optimizer.step()
    return total_loss / len(train_dataset)


@torch.no_grad()
def test(loader):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        pred = model(data.x, data.pos, data.batch).max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)


if __name__ == '__main__':
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data/ModelNet10')
    pre_transform, transform = T.NormalizeScale(), T.SamplePoints(1024)
    train_dataset = ModelNet(path, '10', True, transform, pre_transform)
    test_dataset = ModelNet(path, '10', False, transform, pre_transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                              num_workers=6)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False,
                             num_workers=6)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print(f'The classification model:\n{model}')
    print(f'Parameters: {count_parameters(model)}.\n')
    stats = defaultdict(list)
    start = time.time()

    for epoch in range(1, 201):
        loss = train()
        t_test = time.time()
        test_acc = test(test_loader)
        print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, Test: {test_acc:.4f}')
        stats['train_loss'].append(loss)
        stats['test_acc'].append(test_acc)
        stats['elapsed_t'].append(time.time() - start)
        stats['individual_t_test'].append(time.time() - t_test)

    elapsed = time.time() - start
    elapsed_test = np.sum(stats['individual_t_test'])
    print(f'Elapsed time (total): {elapsed:0.1f}s')
    print(f'Elapsed time (test): {elapsed_test:0.1f}s')

    with open('pointnet2_classification_stats.pkl', 'wb') as fh:
        pickle.dump(stats, fh)

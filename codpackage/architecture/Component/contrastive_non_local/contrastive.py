import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
import numpy as np

class NTXentLoss(nn.Module):
    def __init__(self, temperature, use_cosine_similarity):
        super().__init__()
        self.temperature = temperature
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(torch.cuda.current_device())

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        self.batch_size = zis.shape[0]
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)

        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to(torch.cuda.current_device()).long()
        loss = self.criterion(logits, labels)

        return loss.squeeze()


class Projection(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.first = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride = 1, padding = 1),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        point_x = self.first(x)

        return point_x.squeeze()

class MultiViewNonLocalFeatureContrastiveLoss(nn.Module):
    def __init__(self, n_views, in_channels, temperature, use_cosine_similarity):
        super().__init__()
        self.n_views = n_views
        self.in_channels = in_channels
        self.temperature = temperature
        self.use_cosine_similarity = use_cosine_similarity
        self.projections = nn.ModuleList([
            Projection(self.in_channels) for _ in range(self.n_views)
        ])
        self.linears = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.in_channels, self.in_channels),
                nn.Linear(self.in_channels, 256)
            )
            for _ in range(self.n_views)
        ])

        self.pair_loss = NTXentLoss(self.temperature, self.use_cosine_similarity)
    def forward(self, *args):
        assert len(args) == self.n_views, 'inconsistent length between initialization and forwarding'
        # n view -> (2, n) similarities -> (2, n) positive pair loss
        losses = [ ]
        embeddings = [ self.linears[i](self.projections[i](args[i])) for i in range(self.n_views) ]
        for i in range(self.n_views):
            for j in range(i + 1, self.n_views):
                loss = self.pair_loss(embeddings[i], embeddings[j])
                losses.append(loss.unsqueeze(dim = 0))
        total_loss = torch.cat(losses, dim = 0).sum()

        return total_loss
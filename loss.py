from argparse import Namespace

import torch
import torch.nn.functional as F
from torch import nn

from Margin.miner import DistanceWeightedMiner


class GradRev(torch.autograd.Function):
    """
    Implements an autograd class to flip gradients during backward pass.
    """

    @staticmethod
    def forward(ctx, x):
        """
        Container which applies a simple identity function.

        Input:
            x: any torch tensor input.
        """
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Container to reverse gradient signal during backward pass.

        Input:
            grad_output: any computed gradient.
        """
        return grad_output * -1.0


class MutualLoss(nn.Module):
    def __init__(self, args: Namespace):
        """
        Adversial Loss Function that uses a projection network to decorrelate two embeddings living in
        DIFFERENT embedding spaces. While the projection network learns to closely project both embeddings,
        the gradient reversal ensures that the embeddings are actually decorrelated.
        """
        super().__init__()
        cls_dim, aux_dim, adv_dim = args.n_bits, args.n_bits, args.adv_dim
        # Projection network
        self.regressor = nn.Sequential(nn.Linear(aux_dim, adv_dim), nn.ReLU(), nn.Linear(adv_dim, cls_dim))
        # self.regressor = nn.Identity()

    def forward(self, cls_features, aux_features):
        # Apply gradient reversal on input embeddings.
        features = [
            F.normalize(GradRev.apply(cls_features), dim=-1),
            F.normalize(GradRev.apply(aux_features), dim=-1),
        ]
        # Project one embedding to the space of the other (with normalization).
        features[1] = F.normalize(self.regressor(features[1]), dim=-1)
        # Then compute the correlation.
        loss = -1.0 * ((features[0] * features[1]) ** 2).mean()  # paper
        # loss = -1.0 * (features[0] * features[1]).sum(dim=1).mean() # mine
        return loss


class MarginLoss(nn.Module):
    def __init__(self, args: Namespace, usage="cls"):
        super().__init__()

        self.margin = args.margin

        n_classes = args.n_classes if usage == "cls" else args.n_clusters
        self.beta = nn.Parameter(torch.ones(n_classes) * args.beta)

        self.miner = DistanceWeightedMiner(args)

    def forward(self, dist_mat, labels):
        anc_idxes, pos_idxes, neg_idxes = self.miner(dist_mat.detach(), labels)

        if len(anc_idxes) == 0:
            # print("no triplets")
            return 0, 0

        d_ap = dist_mat[anc_idxes, pos_idxes]
        d_an = dist_mat[anc_idxes, neg_idxes]

        anchor_labels = labels[anc_idxes]

        if labels.ndim == 2:
            beta = torch.einsum("nc,c->n", anchor_labels, self.beta) / anchor_labels.sum(dim=1)
        else:
            # for cluster labels
            beta = self.beta[anchor_labels.to(int)]

        pos_loss = F.relu(d_ap - beta + self.margin)
        neg_loss = F.relu(beta - d_an + self.margin)

        pair_count = torch.sum((pos_loss > 0.0) + (neg_loss > 0.0))

        # Actual Margin Loss
        loss = torch.sum(pos_loss + neg_loss) if pair_count == 0.0 else torch.sum(pos_loss + neg_loss) / pair_count

        return loss, len(anc_idxes)

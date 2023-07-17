from __future__ import print_function

import csv
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


def softXEnt(input, target):
    logprobs = torch.nn.functional.log_softmax(input, dim=1)
    return -(target * logprobs).sum() / input.shape[0]


def Contrastive_Loss(x, y, temperature=750.0):
    batch_size = x.size()[0]
    x = nn.functional.normalize(x, dim=1)
    y = nn.functional.normalize(y, dim=1)
    LARGE_NUM = 1e9
    mask = torch.eye(batch_size)
    label = nn.functional.one_hot(
        torch.arange(0, batch_size), num_classes=2 * batch_size
    ).type(torch.float32)
    if torch.cuda.is_available():
        mask = mask.cuda()
        label = label.cuda()
    logits_aa = torch.matmul(x, x.t()) / temperature
    logits_bb = torch.matmul(y, y.t()) / temperature
    logits_ab = torch.matmul(x, y.t()) / temperature
    logits_ba = torch.matmul(y, x.t()) / temperature

    logits_aa = logits_aa - mask * LARGE_NUM
    logits_bb = logits_bb - mask * LARGE_NUM

    loss_a = softXEnt(torch.cat((logits_ab, logits_aa), dim=1), label)
    loss_b = softXEnt(torch.cat((logits_ba, logits_bb), dim=1), label)

    loss = loss_a + loss_b
    return loss


def calc_neg_mask(target1, target2):
    mask = torch.zeros((target1.size(0), target2.size(0)))
    for i in range(target1.size(0)):
        for j in range(target2.size(0)):
            if target1[i] == target2[j]:
                mask[i][j] = 1
    return mask


def calc_pos_mask(target1, target2):
    mask = torch.zeros((target1.size(0), target2.size(0)))
    for i in range(target1.size(0)):
        for j in range(target2.size(0)):
            if target1[i] != target2[j]:
                mask[i][j] = 1
    return mask


class M_CosineLoss(torch.nn.Module):
    def __init__(self, margin, a=1.0, b=100.0):
        super(M_CosineLoss, self).__init__()
        self.cosine_margin = margin
        self.positive_weight = b
        self.negative_weight = a

    def forward(self, X, Y, t1, t2):
        x_norm = F.normalize(X)
        y_norm = F.normalize(Y)
        batch_size = X.size(0)
        Z = torch.matmul(x_norm, y_norm.t())
        Z = (Z - torch.min(Z).detach()) / (
            torch.max(Z).detach() - torch.min(Z).detach()
        )
        pos_mask = torch.zeros((batch_size, batch_size))
        neg_mask = torch.zeros((batch_size, batch_size))
        for i in range(batch_size):
            for j in range(batch_size):
                if t1[i] == t2[j]:
                    pos_mask[i][j] = 1
                else:
                    neg_mask[i][j] = 1
        if torch.cuda.is_available():
            pos_mask = pos_mask.cuda()
            neg_mask = neg_mask.cuda()
        positive_values = torch.mul(pos_mask, Z)
        negative_values = torch.mul(neg_mask, Z)
        s1 = torch.sum(positive_values)  # sum of positives
        s2 = torch.sum(negative_values)  # sum of negatives
        print(
            "+ve = ",
            self.positive_weight / (s1.item() * (batch_size**2)),
            "-ve = ",
            self.negative_weight * s2.item() / (batch_size**2),
            "\n",
        )
        loss = (self.positive_weight / s1 + self.negative_weight * s2) / (
            batch_size**2
        )

        f = open("pos_neg_mcosine_exp.csv", "a+")
        csv_writer = csv.writer(f)
        csv_element = [
            self.negative_weight * s2.item() / (batch_size**2),
            self.positive_weight / (s1.item() * (batch_size**2)),
        ]
        csv_writer.writerow(csv_element)
        f.close()

        return loss


class E_CosineLoss(torch.nn.Module):
    def __init__(self, margin, a=1.0, b=100.0):
        super(E_CosineLoss, self).__init__()
        self.cosine_margin = margin
        self.positive_weight = b
        self.negative_weight = a

    def forward(self, X, Y, t1, t2):
        x_norm = F.normalize(X)
        y_norm = F.normalize(Y)
        batch_size = X.size(0)
        Z = torch.matmul(x_norm, y_norm.t())
        Z = torch.exp(Z) / torch.sum(torch.exp(Z), dim=1, keepdims=True).detach()
        pos_mask = torch.zeros((batch_size, batch_size))
        neg_mask = torch.zeros((batch_size, batch_size))
        for i in range(batch_size):
            for j in range(batch_size):
                if t1[i] == t2[j]:
                    pos_mask[i][j] = 1
                else:
                    neg_mask[i][j] = 1
        if torch.cuda.is_available():
            pos_mask = pos_mask.cuda()
            neg_mask = neg_mask.cuda()
        positive_values = torch.mul(pos_mask, Z)
        negative_values = torch.mul(neg_mask, Z)
        positive_sum = torch.sum(positive_values)  # sum of positives
        negative_sum = torch.sum(negative_values)  # sum of negatives
        print(
            "+ve loss = ",
            self.positive_weight / (positive_sum * (batch_size**2)),
            "\n-ve loss = ",
            self.negative_weight * negative_sum / (batch_size**2),
            "\n",
        )
        loss = (
            self.positive_weight / positive_sum + self.negative_weight * negative_sum
        ) / (batch_size**2)
        return loss


class LE_CosineLoss(torch.nn.Module):
    def __init__(self, margin, a=1.0, b=0.01):
        super(LE_CosineLoss, self).__init__()
        self.cosine_margin = margin
        self.positive_weight = b
        self.negative_weight = a

    def forward(self, X, Y, t1, t2):
        x_norm = F.normalize(X)
        y_norm = F.normalize(Y)
        batch_size = X.size(0)
        Z = torch.matmul(x_norm, y_norm.t()).type(torch.double)
        Z = torch.exp(Z) / torch.sum(torch.exp(Z), dim=1, keepdims=True).detach()
        pos_mask = torch.zeros((batch_size, batch_size), dtype=torch.int32)
        neg_mask = torch.zeros((batch_size, batch_size), dtype=torch.int32)
        for i in range(batch_size):
            for j in range(batch_size):
                if t1[i] == t2[j]:
                    pos_mask[i][j] = 1
                else:
                    neg_mask[i][j] = 1
        if torch.cuda.is_available():
            pos_mask = pos_mask.cuda()
            neg_mask = neg_mask.cuda()
        positive_values = torch.where(pos_mask == 0, 1.0, Z)
        negative_values = torch.mul(neg_mask, Z)
        positive_loss = -torch.sum(torch.log(positive_values))  # sum of positives
        negative_loss = -torch.sum(torch.log(1 - negative_values))  # sum of negatives
        print(
            "+ve loss = ",
            self.positive_weight * positive_loss.item() / (batch_size**2),
        )
        print(
            "-ve loss = ",
            self.negative_weight * negative_loss.item() / (batch_size**2),
        )
        loss = (
            self.positive_weight * positive_loss + self.negative_weight * negative_loss
        ) / (batch_size**2)
        f = open("pos_neg_lecosine_exp.csv", "a+")
        csv_writer = csv.writer(f)
        csv_element = [
            self.negative_weight * negative_loss.item() / (batch_size**2),
            self.positive_weight * positive_loss.item() / (batch_size**2),
        ]
        csv_writer.writerow(csv_element)
        f.close()
        return loss


class DistanceLoss(torch.nn.Module):
    def __init__(self, margin=1.5, a=1, b=0.01):
        super(DistanceLoss, self).__init__()
        self.euclidean_margin = margin
        self.negative_weight = a
        self.positive_weight = b

    def forward(self, X, Y, t1, t2):
        assert (
            X.size() == Y.size()
        ), "Both the hidden tensors need to be of the same shape."
        batch_size = X.size(0)
        x = F.normalize(X)
        y = F.normalize(Y)
        pos_mask = torch.zeros((batch_size, batch_size))
        neg_mask = torch.zeros((batch_size, batch_size))
        margin_matrix = torch.ones((batch_size, batch_size)) * self.euclidean_margin
        for i in range(batch_size):
            for j in range(batch_size):
                if t1[i] == t2[j]:
                    pos_mask[i][j] = 1
                else:
                    neg_mask[i][j] = 1
        if torch.cuda.is_available():
            pos_mask = pos_mask.cuda()
            neg_mask = neg_mask.cuda()
            margin_matrix = margin_matrix.cuda()

        euclidean_distance = torch.cdist(x, y)
        positive_values = torch.mul(pos_mask, euclidean_distance)
        negative_values = torch.mul(neg_mask, euclidean_distance)

        margin_matrix = torch.mul(neg_mask, margin_matrix)
        positive_loss = torch.sum(torch.square(positive_values))
        negative_loss = torch.sum(torch.square(F.relu(margin_matrix - negative_values)))

        total_loss = (
            self.negative_weight * negative_loss + self.positive_weight * positive_loss
        ) / (batch_size**2)
        # print("-ve loss = ", self.negative_weight*negative_loss.item()/(batch_size**2))
        # print("+ve loss = ", self.positive_weight*positive_loss.item()/(batch_size**2))
        f = open("pos_neg_dist_exp.csv", "a+")
        csv_writer = csv.writer(f)
        csv_element = [
            self.negative_weight * negative_loss.item() / (batch_size**2),
            self.positive_weight * positive_loss.item() / (batch_size**2),
        ]
        csv_writer.writerow(csv_element)
        f.close()
        return total_loss


class AMCLoss(torch.nn.Module):
    def __init__(self, margin, a=10.0, b=0.1):
        super(AMCLoss, self).__init__()
        self.amc_margin = margin
        self.negative_weight = a
        self.positive_weight = b

    def forward(self, x, y, t1, t2):
        assert (
            x.size() == y.size()
        ), "Both the hidden tensors need to be of the same shape."
        batch_size = x.size(0)

        x_norm = F.normalize(x)
        y_norm = F.normalize(y)

        pos_mask = torch.zeros((batch_size, batch_size))
        neg_mask = torch.zeros((batch_size, batch_size))
        margin_matrix = torch.ones((batch_size, batch_size)) * self.amc_margin
        for i in range(batch_size):
            for j in range(batch_size):
                if t1[i] == t2[j]:
                    pos_mask[i][j] = 1
                else:
                    neg_mask[i][j] = 1
        if torch.cuda.is_available():
            pos_mask = pos_mask.cuda()
            neg_mask = neg_mask.cuda()
            margin_matrix = margin_matrix.cuda()

        geodesic_distance = torch.acos(
            torch.clip(torch.matmul(x_norm, y_norm.t()), -1.0 + 1e-7, 1.0 - 1e-7)
        )
        positive_values = torch.mul(pos_mask, geodesic_distance)
        negative_values = torch.mul(neg_mask, geodesic_distance)

        margin_matrix = torch.mul(neg_mask, margin_matrix)
        positive_loss = torch.sum(torch.square(positive_values))
        negative_loss = torch.sum(torch.square(F.relu(margin_matrix - negative_values)))

        total_loss = (
            self.negative_weight * negative_loss + self.positive_weight * positive_loss
        ) / (batch_size**2)
        print(
            "-ve loss = ",
            self.negative_weight * negative_loss.item() / (batch_size**2),
        )
        print(
            "+ve loss = ",
            self.positive_weight * positive_loss.item() / (batch_size**2),
        )

        f = open("pos_neg_amc_exp.csv", "a+")
        csv_writer = csv.writer(f)
        csv_element = [
            self.negative_weight * negative_loss.item() / (batch_size**2),
            self.positive_weight * positive_loss.item() / (batch_size**2),
        ]
        csv_writer.writerow(csv_element)
        f.close()

        return total_loss


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, contrast_mode="all", base_temperature=0.07):
        super(SupConLoss, self).__init__()
        # self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, temperature=0.07, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = torch.device("cuda") if features.is_cuda else torch.device("cpu")

        if len(features.shape) < 3:
            raise ValueError(
                "`features` needs to be [bsz, n_views, ...],"
                "at least 3 dimensions are required"
            )
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`")
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError("Unknown mode: {}".format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), temperature
        )
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0,
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mask_sum = mask.sum(1)
        mask_sum[mask_sum == 0] += 1

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum

        # loss
        loss = -(temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

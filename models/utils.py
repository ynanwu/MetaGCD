import torch
from torch.nn import functional as F
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm
from project_utils.cluster_and_log_utils import log_accs_from_preds
from data.data_utils import re_assign_labels
import torch.nn.functional as F

class SupConLoss(torch.nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR
    From: https://github.com/HobbitLong/SupContrast"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None, weight=None, device=None):
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

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)
            weight = weight.to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

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
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        if weight is not None:
            weight = weight.repeat(anchor_count, contrast_count)
            weight = weight * logits_mask
            mean_log_prob_pos = (weight * mask * log_prob).sum(1) / mask.sum(1)
        else:
            mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


def info_nce_logits(features, args, device):

    b_ = 0.5 * int(features.size(0))

    labels = torch.cat([torch.arange(b_) for i in range(args.n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(device)

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)
    # assert similarity_matrix.shape == (
    #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
    # assert similarity_matrix.shape == labels.shape

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

    logits = logits / args.temperature
    return logits, labels


def test_kmeans(model, test_loader,  stage,
                epoch, save_name,
                args, device):

    model.eval()

    all_feats = []
    targets = np.array([])
    mask = np.array([])

    # print('Collating features...')
    # First extract all features
    for batch_idx, (images, label, _) in enumerate(test_loader):
        # print('test_kmeans batch_idx:', batch_idx)
        images = images.to(device)
        # Pass features through base model and then additional learnable transform (linear layer)
        feats = model(images)

        feats = torch.nn.functional.normalize(feats, dim=-1)

        all_feats.append(feats.cpu().numpy())
        targets = np.append(targets, label.cpu().numpy())
        mask = np.append(mask, np.array([True if x.item() in args.train_classes
                                         else False for x in label]))

    # K-MEANS
    # hierarchical_print(f'Fitting K-Means', length=line_length, mode='in', level=2)
    all_feats = np.concatenate(all_feats)
    n_clusters = args.num_labeled_classes + args.num_cur_unlabeled_classes
    # n_clusters = cluster_num(75, 105, all_feats)
    # print('best_n_clusters:', n_clusters)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(all_feats)
    # kmeans = KMeans(n_clusters=args.num_labeled_classes, random_state=0).fit(all_feats)
    preds = kmeans.labels_

    targets = re_assign_labels(targets)

    # -----------------------
    # EVALUATE
    # -----------------------
    if stage == 'pretrain':
        old_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask, stage='pretrain',
                                                        T=epoch, eval_funcs=args.eval_funcs, save_name=save_name)
        return old_acc
    else:
        all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask, stage='metaTest',
                                                        T=epoch, eval_funcs=args.eval_funcs, save_name=save_name)

        return all_acc, old_acc, new_acc

def meta_update_model(model, optimizer, loss, gradients):
    # Hack from https://github.com/yiweilu3/Few-shot-Scene-adaptive-Anomaly-Detection/blob/8f69aa847514538ac1169264d5dc545ab9a59505/train.py#L52
    # Register a hook on each parameter in the net that replaces the current dummy grad
    # with our grads accumulated across the meta-batch
    hooks = []
    for (k, v) in model.named_parameters():
        def get_closure():
            key = k

            def replace_grad(grad):
                return gradients[key]

            return replace_grad

        hooks.append(v.register_hook(get_closure()))

    # Compute grads for current step, replace with summed gradients as defined by hook
    optimizer.zero_grad()

    loss.backward()

    # Update the net parameters with the accumulated gradient according to optimizer
    optimizer.step()

    # Remove the hooks before next training phase
    for h in hooks:
        h.remove()

def generate_pseudo_labels(feats, k_threshold=0.80):
    batch_num = int(feats.shape[0]/2)
    dist = F.cosine_similarity(feats[:, :, None], feats.t()[None, :, :])
    d1, d2 = [d for d in dist.chunk(2)]
    dist = torch.maximum(d1, d2)
    dist = torch.maximum(dist[:, :batch_num], dist[:, batch_num:])
    # threshold = dist.topk(topk)[0][:, -1].unsqueeze(1)
    dist[dist < k_threshold] = 0
    dist[dist >= k_threshold] = 1

    simi = torch.matmul(feats, feats.T)
    # simi = (simi - simi.min()) / (simi.max() - simi.min())
    s1, s2 = [s for s in simi.chunk(2)]
    simi = torch.maximum(s1, s2)
    simi = torch.maximum(simi[:, :batch_num], simi[:, batch_num:])
    simi[~(dist.bool())] = 0

    w_attention = (simi+(dist-1)*99999).softmax(dim=1)
    w_attention = w_attention / w_attention.max(dim=1)[0].unsqueeze(1) #256x256

    pseudo_mask = dist.sum(dim=1)>1 #256
    simi_mask = dist[pseudo_mask]
    w_attention = w_attention[pseudo_mask]
    return simi_mask[:, pseudo_mask], pseudo_mask, w_attention[:, pseudo_mask]


def Silhouette_ALL(n, feats):
    # feats = feats.detach().cpu().numpy()
    data_Cluster = KMeans(n_clusters=n, random_state=0).fit(feats)
    label = data_Cluster.labels_
    Silhouette_Coefficient = silhouette_score(feats, label)
    return Silhouette_Coefficient

def cluster_num(lower_bound, upper_bound, feats):
    coefficient = []
    for k in range(lower_bound, upper_bound):
        print('Silhouette_k:', k)
        data_data_Silhouette_mean = Silhouette_ALL(k, feats)
        coefficient.append(data_data_Silhouette_mean)
    clus_num = list(range(lower_bound, upper_bound))[coefficient.index(max(coefficient))]
    return clus_num

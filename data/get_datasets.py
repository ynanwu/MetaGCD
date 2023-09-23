import torch

from data.data_utils import MergedDataset, MergedUnlabelledDataset
#[1: ]
from data.cifar import get_cifar_10_datasets, get_cifar_100_datasets
from data.imagenet import get_imagenet_100_datasets

from data.cifar import subsample_classes as subsample_dataset_cifar
from data.imagenet import subsample_classes as subsample_dataset_imagenet

from copy import deepcopy
from tqdm import tqdm

sub_sample_class_funcs = {
    'cifar10': subsample_dataset_cifar,
    'cifar100': subsample_dataset_cifar,
    'imagenet_100': subsample_dataset_imagenet,
}

get_dataset_funcs = {
    'cifar10': get_cifar_10_datasets,
    'cifar100': get_cifar_100_datasets,
    'imagenet_100': get_imagenet_100_datasets,
}

def get_datasets(dataset_name, train_transform, test_transform, args):
    """
    :return: train_dataset: MergedDataset which concatenates labelled and unlabelled
             test_dataset,
             unlabelled_train_examples_test,
             datasets
    """
    if dataset_name not in get_dataset_funcs.keys():
        raise ValueError

    # Get datasets
    get_dataset_f = get_dataset_funcs[dataset_name]
    datasets = get_dataset_f(train_transform=train_transform, test_transform=test_transform,
                             train_classes=args.train_classes,
                             prop_train_labels=args.prop_train_labels,
                             split_train_val=False)

    # Set target transforms:
    target_transform_dict = {}
    for i, cls in enumerate(list(args.train_classes) + list(args.unlabeled_classes)):
        target_transform_dict[cls] = i
    target_transform = lambda x: target_transform_dict[x]

    for dataset_name, dataset in datasets.items():
        if dataset is not None:
            if type(dataset) is list:
                for d in dataset:
                    d.target_transform = target_transform
            else:
                dataset.target_transform = target_transform

    # === pretrain dataset ===
    pretrain_train_dataset = MergedUnlabelledDataset(old_unlabelled_dataset=deepcopy(datasets['pretrain_pseudo_old_cls_train_samples']),
                                  novel_unlabelled_dataset=[], pretrain=True)
    pretrain_test_dataset = datasets['pretrain_pseudo_old_cls_test_samples']

    # === offline session data ===
    offline_session_train_dataset_list = []
    for offline_old_dataset_unlabelled, offline_novel_dataset_unlabelled in zip(
            datasets['offline_pseudo_old_unlabelled_list'], datasets['offline_pseudo_novel_unlabelled_list']):
        online_session_train_dataset = MergedUnlabelledDataset(
            old_unlabelled_dataset=deepcopy(offline_old_dataset_unlabelled),
            novel_unlabelled_dataset=deepcopy(offline_novel_dataset_unlabelled))
        offline_session_train_dataset_list.append(online_session_train_dataset)

    offline_session_test_dataset_list = datasets['offline_test_dataset_list']

    # === online session data ===
    online_session_train_dataset_list = []
    for old_dataset_unlabelled, novel_dataset_unlabelled in zip(
            datasets['online_old_dataset_unlabelled_list'], datasets['online_novel_dataset_unlabelled_list']):
        online_session_train_dataset = MergedUnlabelledDataset(
            old_unlabelled_dataset=deepcopy(old_dataset_unlabelled),
            novel_unlabelled_dataset=deepcopy(novel_dataset_unlabelled))
        online_session_train_dataset_list.append(online_session_train_dataset)

    online_session_test_dataset_list = datasets['online_test_dataset_list']

    return pretrain_train_dataset, pretrain_test_dataset, offline_session_train_dataset_list, offline_session_test_dataset_list,\
        online_session_train_dataset_list, online_session_test_dataset_list, datasets

def get_class_splits(args):
    # -------------
    # GET CLASS SPLITS
    # -------------
    if args.dataset_name == 'cifar10':
        args.image_size = 32
        args.train_classes = range(7)
        args.unlabeled_classes = range(7, 10)

    elif args.dataset_name == 'cifar100':
        args.image_size = 32
        args.train_classes = range(80)
        args.unlabeled_classes = range(80, 100)

    elif args.dataset_name == 'tinyimagenet':
        args.image_size = 64
        args.train_classes = range(100)
        args.unlabeled_classes = range(100, 200)
    else:
        raise NotImplementedError

    return args


def Generate_Mean_Var(model, labeled_train_loader, classes_id, device):
    all_feat = []
    all_labels = []

    for epoch in range(1):
        model.eval()   #x:128x3x32x32
        for batch_idx, (images, label, _) in enumerate(tqdm(labeled_train_loader)):
            images, label = images.to(device), label.to(device)
            feats = torch.nn.functional.normalize(model(images), dim=-1)
            all_feat.append(feats.detach().clone().cuda())
            all_labels.append(label.detach().clone().cuda())

    all_feat = torch.cat(all_feat, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    class_mean = torch.stack([torch.mean(all_feat[all_labels == cls], dim=0) for cls in classes_id], dim=0)
    class_var = torch.stack([(torch.var(all_feat[all_labels == cls], dim=0) + 1e-5).sqrt() for cls in classes_id], dim=0)

    return class_mean, class_var


def replay_old_features(class_mean, class_var, args):
    feats = []

    if args.dataset_name == 'cifar10':
        num_per_class = args.replay_num_per_class  #20
    elif args.dataset_name == 'cifar100':
        num_per_class = args.replay_num_per_class  #2
    else:
        num_per_class = 3

    for i in range(args.num_replay_classes):
        dist = torch.distributions.Normal(class_mean[i], class_var.mean(dim=0))
        this_feat = dist.sample((num_per_class,)) #20x512
        feats.append(this_feat)

    feats = torch.stack([i[j] for j in range(num_per_class) for i in feats])
    view1_feats, view2_feats = [f for f in feats.chunk(2)]
    labels = torch.tensor(list(range(args.num_replay_classes))).repeat(int(num_per_class/2))

    return view1_feats, view2_feats, labels

class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for i in range(self.n_views)]
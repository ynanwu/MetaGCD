from torchvision.datasets import CIFAR10, CIFAR100
from copy import deepcopy
import numpy as np

from data.data_utils import subsample_instances
from config import cifar_10_root, cifar_100_root

dataset_split_config_dict = {
    'cifar100': {'pseudo_old_cls_num': 60,
                 'pseudo_novel_cls_num': 20,
                 'pseudo_test_sample_ratio': 0.25,
                 'pseudo_continual_session_num': 4,
                 'pseudo_each_session_novel_sample_num': 200,
                 'test_continual_session_num': 4,
                 'test_each_session_novel_sample_num': 300,
                 },
    'cifar10': {'pseudo_old_cls_num': 4,
                 'pseudo_novel_cls_num': 3,
                 'pseudo_test_sample_ratio': 0.25,
                 'pseudo_continual_session_num': 3,
                 'pseudo_each_session_novel_sample_num': 2000,
                 'test_continual_session_num': 3,
                 'test_each_session_novel_sample_num': 2500,
                 },
}

class CustomCIFAR10(CIFAR10):

    def __init__(self, *args, **kwargs):
        super(CustomCIFAR10, self).__init__(*args, **kwargs)

        self.uq_idxs = np.array(range(len(self)))

    def __getitem__(self, item):
        img, label = super().__getitem__(item)
        uq_idx = self.uq_idxs[item]

        return img, label, uq_idx

    def __len__(self):
        return len(self.targets)


class CustomCIFAR100(CIFAR100):

    def __init__(self, *args, **kwargs):
        super(CustomCIFAR100, self).__init__(*args, **kwargs)

        self.uq_idxs = np.array(range(len(self)))

    def __getitem__(self, item):
        img, label = super().__getitem__(item)
        uq_idx = self.uq_idxs[item]

        return img, label, uq_idx

    def __len__(self):
        return len(self.targets)


def subsample_dataset(dataset, idxs):
    # Allow for setting in which all empty set of indices is passed

    if len(idxs) > 0:

        dataset.data = dataset.data[idxs]
        dataset.targets = np.array(dataset.targets)[idxs].tolist()
        dataset.uq_idxs = dataset.uq_idxs[idxs]

        return dataset

    else:

        return None


def subsample_classes(dataset, include_classes=(0, 1, 8, 9)):
    cls_idxs = [x for x, t in enumerate(dataset.targets) if t in include_classes]

    target_xform_dict = {}
    for i, k in enumerate(include_classes):
        target_xform_dict[k] = i

    dataset = subsample_dataset(dataset, cls_idxs)

    # dataset.target_transform = lambda x: target_xform_dict[x]

    return dataset

def subDataset_wholeDataset(datalist):
    wholeDataset = deepcopy(datalist[0])
    wholeDataset.data = np.concatenate([
        d.data for d in datalist], axis=0)
    wholeDataset.targets = np.concatenate([
        d.targets for d in datalist], axis=0).tolist()
    wholeDataset.uq_idxs = np.concatenate([
        d.uq_idxs for d in datalist], axis=0)

    return wholeDataset

def get_train_val_indices(train_dataset, val_split=0.2):
    train_classes = np.unique(train_dataset.targets)

    # Get train/test indices
    train_idxs = []
    val_idxs = []
    for cls in train_classes:
        cls_idxs = np.where(train_dataset.targets == cls)[0]

        v_ = np.random.choice(cls_idxs, replace=False, size=((int(val_split * len(cls_idxs))),))
        t_ = [x for x in cls_idxs if x not in v_]

        train_idxs.extend(t_)
        val_idxs.extend(v_)

    return train_idxs, val_idxs

def get_cifar_10_datasets_old(train_transform, test_transform, train_classes=(0, 1, 8, 9),
                           prop_train_labels=0.8, split_train_val=False):

    pseudo_test_sample_ratio = dataset_split_config_dict['cifar10']['pseudo_test_sample_ratio']
    test_continual_session_num = dataset_split_config_dict['cifar10']['test_continual_session_num']
    test_each_session_novel_sample_num = dataset_split_config_dict['cifar10']['test_each_session_novel_sample_num']

    # Init entire training set
    whole_training_set = CustomCIFAR10(root=cifar_10_root, transform=train_transform, train=True)

    # Get labelled training set which has subsampled classes, then subsample some indices from that
    old_dataset_all = subsample_classes(deepcopy(whole_training_set), include_classes=train_classes)  # 35000

    each_old_all_samples = [subsample_classes(deepcopy(old_dataset_all), include_classes=[targets])
                            for targets in range(len(list(train_classes)))]  # 7*5000

    each_old_labeled_slices = [subsample_instances(samples, prop_indices_to_subsample=prop_train_labels)
                               for samples in each_old_all_samples]  # 7*4000

    each_old_unlabeled_slices = [
        np.array(list(set(list(range(len(samples.targets)))) - set(each_old_labeled_slices[i])))
        for i, samples in enumerate(each_old_all_samples)]  # 7*1000

    each_old_labeled_samples = [subsample_dataset(deepcopy(samples), each_old_labeled_slices[i])
                                for i, samples in enumerate(each_old_all_samples)]  # 7*4000

    online_old_unlabeled_samples = [subsample_dataset(deepcopy(samples), each_old_unlabeled_slices[i])
                                    for i, samples in
                                    enumerate(each_old_all_samples)]  # 7*1000 for online old classes unlabeled

    offline_test_dataset_slices = [subsample_instances(samples, prop_indices_to_subsample=pseudo_test_sample_ratio)
                                   for samples in each_old_labeled_samples]  # 7*1000 for pseudo test dataset
    offline_train_dataset_slices = [
        np.array(list(set(list(range(len(samples.targets)))) - set(offline_test_dataset_slices[i])))
        for i, samples in enumerate(each_old_labeled_samples)]  # 7*3000

    offline_test_dataset_samples = [subsample_dataset(deepcopy(samples), offline_test_dataset_slices[i])
                                    for i, samples in
                                    enumerate(each_old_labeled_samples)]  # 7*1000 for pseudo test samples
    offline_train_dataset_samples = [subsample_dataset(deepcopy(samples), offline_train_dataset_slices[i])
                                     for i, samples in enumerate(each_old_labeled_samples)]  # 7*3000=24000

    offline_train_dataset_samples = subDataset_wholeDataset(
        [offline_train_dataset_samples[cls] for cls in train_classes])
    offline_test_dataset_samples = subDataset_wholeDataset([offline_test_dataset_samples[cls] for cls in train_classes])
    offline_test_dataset_samples.transform = test_transform

    # ------------------------------online old classes unlabeled samples------------------------------------------------
    online_old_unlabelled_indices_shuffle = np.array(list(range(len(online_old_unlabeled_samples[0]))))
    np.random.shuffle(online_old_unlabelled_indices_shuffle)
    online_old_unlabelled_indices_list = np.array_split(online_old_unlabelled_indices_shuffle,
                                                        test_continual_session_num)

    online_old_dataset_unlabelled_list = [subDataset_wholeDataset([subsample_dataset(deepcopy(sample), indices)
                                                                   for sample in online_old_unlabeled_samples])
                                          for indices in online_old_unlabelled_indices_list]  # list:4x200

    # ------------------------------online novel classes unlabeled samples----------------------------------------------
    novel_unlabelled_indices = set(whole_training_set.uq_idxs) - set(old_dataset_all.uq_idxs) #15000
    novel_dataset_unlabelled = subsample_dataset(deepcopy(whole_training_set),
                                                 np.array(list(novel_unlabelled_indices)))  # 15000
    # Get test set for all classes
    test_dataset = CustomCIFAR10(root=cifar_10_root, transform=test_transform, train=False)  # 10000

    novel_targets_shuffle = np.array(list(set(np.array(novel_dataset_unlabelled.targets).tolist())))
    np.random.shuffle(novel_targets_shuffle)

    online_novel_dataset_unlabelled_list = []
    online_test_dataset_list = []
    targets_per_session = len(novel_targets_shuffle) // test_continual_session_num
    for s in range(test_continual_session_num):
        online_session_targets = novel_targets_shuffle[0: s * targets_per_session + targets_per_session]
        online_session_novel_samples = [subsample_classes(deepcopy(novel_dataset_unlabelled), include_classes=[targets])
                                        for targets in online_session_targets]
        online_session_novel_samples = [subsample_dataset(deepcopy(samples),  # random sample
                                                          np.random.choice(
                                                              np.array(list(range(len(samples.targets)))),
                                                              test_each_session_novel_sample_num, replace=False))
                                        for samples in online_session_novel_samples]

        online_session_novel_dataset = subDataset_wholeDataset(online_session_novel_samples)
        online_novel_dataset_unlabelled_list.append(online_session_novel_dataset)
        online_session_test_dataset = subsample_classes(
            deepcopy(test_dataset), include_classes=list(train_classes) + online_session_targets.tolist())
        online_test_dataset_list.append(online_session_test_dataset)

    all_datasets = {
        'offline_train_dataset': offline_train_dataset_samples,  # 21000
        'offline_test_dataset': offline_test_dataset_samples,  # 7000
        'online_old_dataset_unlabelled_list': online_old_dataset_unlabelled_list,  # list 4x2000
        'online_novel_dataset_unlabelled_list': online_novel_dataset_unlabelled_list,  # list4:(2500,5000,7500)
        'online_test_dataset_list': online_test_dataset_list,  # list4:(8000,9000,10000)
    }
    return all_datasets


def get_cifar_10_datasets(train_transform, test_transform, train_classes=(0, 1, 8, 9),
                           prop_train_labels=0.8, split_train_val=False):

    pseudo_old_cls_num = dataset_split_config_dict['cifar10']['pseudo_old_cls_num']
    pseudo_test_sample_ratio = dataset_split_config_dict['cifar10']['pseudo_test_sample_ratio']
    pseudo_continual_session_num = dataset_split_config_dict['cifar10']['pseudo_continual_session_num']
    pseudo_each_session_novel_sample_num = dataset_split_config_dict['cifar10']['pseudo_each_session_novel_sample_num']
    test_continual_session_num = dataset_split_config_dict['cifar10']['test_continual_session_num']
    test_each_session_novel_sample_num = dataset_split_config_dict['cifar10']['test_each_session_novel_sample_num']

    # Init entire training set
    whole_training_set = CustomCIFAR10(root=cifar_10_root, transform=train_transform, train=True)

    # Get labelled training set which has subsampled classes, then subsample some indices from that
    old_dataset_all = subsample_classes(deepcopy(whole_training_set), include_classes=train_classes)

    each_old_all_samples = [subsample_classes(deepcopy(old_dataset_all), include_classes=[targets])
                                                                for targets in range(len(list(train_classes)))]

    each_old_labeled_slices = [subsample_instances(samples, prop_indices_to_subsample=prop_train_labels)
                                                                for samples in each_old_all_samples]
    each_old_unlabeled_slices = [np.array(list(set(list(range(len(samples.targets))))-set(each_old_labeled_slices[i])))
                                                                for i, samples in enumerate(each_old_all_samples)]

    each_old_labeled_samples = [subsample_dataset(deepcopy(samples), each_old_labeled_slices[i])
                                                                for i, samples in enumerate(each_old_all_samples)]

    online_old_unlabeled_samples = [subsample_dataset(deepcopy(samples), each_old_unlabeled_slices[i])
                                                                 for i, samples in enumerate(each_old_all_samples)]

    offline_test_dataset_slices = [subsample_instances(samples, prop_indices_to_subsample=pseudo_test_sample_ratio)
                                                                 for samples in each_old_labeled_samples]
    offline_train_dataset_slices = [np.array(list(set(list(range(len(samples.targets))))-set(offline_test_dataset_slices[i])))
                                                                for i, samples in enumerate(each_old_labeled_samples)]

    offline_test_dataset_samples = [subsample_dataset(deepcopy(samples), offline_test_dataset_slices[i])
                                                                for i, samples in enumerate(each_old_labeled_samples)]
    offline_train_dataset_samples = [subsample_dataset(deepcopy(samples), offline_train_dataset_slices[i])
                                                                for i, samples in enumerate(each_old_labeled_samples)]

    old_ids_all = list(train_classes)
    np.random.shuffle(old_ids_all)

    pseudo_old_cls = old_ids_all[:pseudo_old_cls_num]
    pseudo_novel_cls = list(set(old_ids_all)-set(pseudo_old_cls))
    np.random.shuffle(pseudo_novel_cls)
    offline_pseudo_old_cls_all_samples = [offline_train_dataset_samples[cls] for cls in pseudo_old_cls]

    offline_pseudo_old_cls_labeled_slices = [np.random.choice(range(len(samples)), replace=False, size=2000) for samples in
                               offline_pseudo_old_cls_all_samples]

    offline_pseudo_old_cls_unlabeled_slices = [np.array(list(set(list(range(len(samples.targets)))) - set(offline_pseudo_old_cls_labeled_slices[i])))
                                             for i, samples in enumerate(offline_pseudo_old_cls_all_samples)]

    pretrain_pseudo_old_cls_train_samples = subDataset_wholeDataset([subsample_dataset(deepcopy(samples), offline_pseudo_old_cls_labeled_slices[i])
                                              for i, samples in enumerate(offline_pseudo_old_cls_all_samples)])
    offline_pseudo_old_cls_unlabeled_samples = [subsample_dataset(deepcopy(samples), offline_pseudo_old_cls_unlabeled_slices[i])
                                              for i, samples in enumerate(offline_pseudo_old_cls_all_samples)]

    pretrain_pseudo_old_cls_test_samples = subDataset_wholeDataset([offline_test_dataset_samples[cls] for cls in pseudo_old_cls])
    pretrain_pseudo_old_cls_test_samples.transform = test_transform

    #------------------------------offline pseudo old classes unlabeled samples-----------------------------------------
    offline_old_unlabelled_indices_shuffle = np.array(list(range(len(offline_pseudo_old_cls_unlabeled_slices[0]))))
    np.random.shuffle(offline_old_unlabelled_indices_shuffle)
    offline_old_unlabelled_indices_list = np.array_split(offline_old_unlabelled_indices_shuffle, pseudo_continual_session_num)

    offline_pseudo_old_unlabelled_list = [subDataset_wholeDataset([subsample_dataset(deepcopy(sample), indices)
                                          for sample in offline_pseudo_old_cls_unlabeled_samples]) for indices in offline_old_unlabelled_indices_list]


    # ------------------------------offline pseudo novel class----------------------------------------------------------
    offline_pseudo_novel_unlabelled_list = []
    offline_test_dataset_list = []
    pseudo_targets_per_session = len(pseudo_novel_cls) // pseudo_continual_session_num
    for s in range(test_continual_session_num):
        offline_session_targets = pseudo_novel_cls[0: s * pseudo_targets_per_session + pseudo_targets_per_session]
        offline_session_novel_samples = [offline_train_dataset_samples[target] for target in offline_session_targets]
        offline_session_novel_samples = [subsample_dataset(deepcopy(samples),  # random sample
                                                          np.random.choice(
                                                              np.array(list(range(len(samples.targets)))),
                                                              pseudo_each_session_novel_sample_num, replace=False))
                                        for samples in offline_session_novel_samples]
        offline_session_novel_dataset = subDataset_wholeDataset(offline_session_novel_samples)
        offline_pseudo_novel_unlabelled_list.append(offline_session_novel_dataset)
        offline_session_test_dataset = subDataset_wholeDataset([offline_test_dataset_samples[cls] for cls in (pseudo_old_cls + offline_session_targets)])
        offline_session_test_dataset.transform = test_transform
        offline_test_dataset_list.append(offline_session_test_dataset)

    # ------------------------------online old classes unlabeled samples------------------------------------------------
    online_old_unlabelled_indices_shuffle = np.array(list(range(len(online_old_unlabeled_samples[0]))))
    np.random.shuffle(online_old_unlabelled_indices_shuffle)
    online_old_unlabelled_indices_list = np.array_split(online_old_unlabelled_indices_shuffle,
                                                         test_continual_session_num)

    online_old_dataset_unlabelled_list = [subDataset_wholeDataset([subsample_dataset(deepcopy(sample), indices)
                                          for sample in online_old_unlabeled_samples]) for indices in
                                          online_old_unlabelled_indices_list]

    # ------------------------------online novel classes unlabeled samples----------------------------------------------
    novel_unlabelled_indices = set(whole_training_set.uq_idxs) - set(old_dataset_all.uq_idxs)
    novel_dataset_unlabelled = subsample_dataset(deepcopy(whole_training_set),
                                                 np.array(list(novel_unlabelled_indices)))
    # Get test set for all classes
    test_dataset = CustomCIFAR10(root=cifar_10_root, transform=test_transform, train=False)

    novel_targets_shuffle = np.array(list(set(np.array(novel_dataset_unlabelled.targets).tolist())))
    np.random.shuffle(novel_targets_shuffle)

    online_novel_dataset_unlabelled_list = []
    online_test_dataset_list = []
    targets_per_session = len(novel_targets_shuffle) // test_continual_session_num
    for s in range(test_continual_session_num):
        online_session_targets = novel_targets_shuffle[0: s * targets_per_session + targets_per_session]
        online_session_novel_samples = [subsample_classes(deepcopy(novel_dataset_unlabelled), include_classes=[targets])
                                        for targets in online_session_targets]
        online_session_novel_samples = [subsample_dataset(deepcopy(samples),  # random sample
                                                       np.random.choice(
                                                           np.array(list(range(len(samples.targets)))),
                                                           test_each_session_novel_sample_num, replace=False))
                                                                for samples in online_session_novel_samples]

        online_session_novel_dataset = subDataset_wholeDataset(online_session_novel_samples)
        online_novel_dataset_unlabelled_list.append(online_session_novel_dataset)
        online_session_test_dataset = subsample_classes(
            deepcopy(test_dataset), include_classes=list(train_classes) + online_session_targets.tolist())
        online_test_dataset_list.append(online_session_test_dataset)

    all_datasets = {
        'pretrain_pseudo_old_cls_train_samples': pretrain_pseudo_old_cls_train_samples,
        'pretrain_pseudo_old_cls_test_samples':pretrain_pseudo_old_cls_test_samples,
        'offline_pseudo_old_unlabelled_list': offline_pseudo_old_unlabelled_list,
        'offline_pseudo_novel_unlabelled_list': offline_pseudo_novel_unlabelled_list,
        'offline_test_dataset_list': offline_test_dataset_list,
        'online_old_dataset_unlabelled_list': online_old_dataset_unlabelled_list,
        'online_novel_dataset_unlabelled_list': online_novel_dataset_unlabelled_list,
        'online_test_dataset_list': online_test_dataset_list,
    }

    return all_datasets

def get_cifar_100_datasets(train_transform, test_transform, train_classes=range(80),
                           prop_train_labels=0.8, split_train_val=False):

    pseudo_old_cls_num = dataset_split_config_dict['cifar100']['pseudo_old_cls_num']
    pseudo_test_sample_ratio = dataset_split_config_dict['cifar100']['pseudo_test_sample_ratio']
    pseudo_continual_session_num = dataset_split_config_dict['cifar100']['pseudo_continual_session_num']
    pseudo_each_session_novel_sample_num = dataset_split_config_dict['cifar100']['pseudo_each_session_novel_sample_num']
    test_continual_session_num = dataset_split_config_dict['cifar100']['test_continual_session_num']
    test_each_session_novel_sample_num = dataset_split_config_dict['cifar100']['test_each_session_novel_sample_num']

    # Init entire training set
    whole_training_set = CustomCIFAR100(root=cifar_100_root, transform=train_transform, train=True)

    # Get labelled training set which has subsampled classes, then subsample some indices from that
    old_dataset_all = subsample_classes(deepcopy(whole_training_set), include_classes=train_classes)

    each_old_all_samples = [subsample_classes(deepcopy(old_dataset_all), include_classes=[targets])
                                                                for targets in range(len(list(train_classes)))]

    each_old_labeled_slices = [subsample_instances(samples, prop_indices_to_subsample=prop_train_labels)
                                                                for samples in each_old_all_samples]
    each_old_unlabeled_slices = [np.array(list(set(list(range(len(samples.targets))))-set(each_old_labeled_slices[i])))
                                                                for i, samples in enumerate(each_old_all_samples)]

    each_old_labeled_samples = [subsample_dataset(deepcopy(samples), each_old_labeled_slices[i])
                                                                for i, samples in enumerate(each_old_all_samples)]

    online_old_unlabeled_samples = [subsample_dataset(deepcopy(samples), each_old_unlabeled_slices[i])
                                                                 for i, samples in enumerate(each_old_all_samples)] #for online old classes unlabeled

    offline_test_dataset_slices = [subsample_instances(samples, prop_indices_to_subsample=pseudo_test_sample_ratio)
                                                                 for samples in each_old_labeled_samples] # for pseudo test dataset
    offline_train_dataset_slices = [np.array(list(set(list(range(len(samples.targets))))-set(offline_test_dataset_slices[i])))
                                                                for i, samples in enumerate(each_old_labeled_samples)]

    offline_test_dataset_samples = [subsample_dataset(deepcopy(samples), offline_test_dataset_slices[i])
                                                                for i, samples in enumerate(each_old_labeled_samples)]  #for pseudo test samples
    offline_train_dataset_samples = [subsample_dataset(deepcopy(samples), offline_train_dataset_slices[i])
                                                                for i, samples in enumerate(each_old_labeled_samples)] #

    old_ids_all = list(train_classes)
    np.random.shuffle(old_ids_all)

    pseudo_old_cls = old_ids_all[:pseudo_old_cls_num]
    pseudo_novel_cls = list(set(old_ids_all)-set(pseudo_old_cls))
    np.random.shuffle(pseudo_novel_cls)
    offline_pseudo_old_cls_all_samples = [offline_train_dataset_samples[cls] for cls in pseudo_old_cls]

    offline_pseudo_old_cls_labeled_slices = [np.random.choice(range(len(samples)), replace=False, size=200) for samples in
                               offline_pseudo_old_cls_all_samples]

    offline_pseudo_old_cls_unlabeled_slices = [np.array(list(set(list(range(len(samples.targets)))) - set(offline_pseudo_old_cls_labeled_slices[i])))
                                             for i, samples in enumerate(offline_pseudo_old_cls_all_samples)]

    pretrain_pseudo_old_cls_train_samples = subDataset_wholeDataset([subsample_dataset(deepcopy(samples), offline_pseudo_old_cls_labeled_slices[i])
                                              for i, samples in enumerate(offline_pseudo_old_cls_all_samples)])
    offline_pseudo_old_cls_unlabeled_samples = [subsample_dataset(deepcopy(samples), offline_pseudo_old_cls_unlabeled_slices[i])
                                              for i, samples in enumerate(offline_pseudo_old_cls_all_samples)]

    pretrain_pseudo_old_cls_test_samples = subDataset_wholeDataset([offline_test_dataset_samples[cls] for cls in pseudo_old_cls])
    pretrain_pseudo_old_cls_test_samples.transform = test_transform

    #------------------------------offline pseudo old classes unlabeled samples-----------------------------------------
    offline_old_unlabelled_indices_shuffle = np.array(list(range(len(offline_pseudo_old_cls_unlabeled_slices[0]))))
    np.random.shuffle(offline_old_unlabelled_indices_shuffle)
    offline_old_unlabelled_indices_list = np.array_split(offline_old_unlabelled_indices_shuffle, pseudo_continual_session_num)

    offline_pseudo_old_unlabelled_list = [subDataset_wholeDataset([subsample_dataset(deepcopy(sample), indices)
                                          for sample in offline_pseudo_old_cls_unlabeled_samples]) for indices in offline_old_unlabelled_indices_list]


    # ------------------------------offline pseudo novel class----------------------------------------------------------
    offline_pseudo_novel_unlabelled_list = []
    offline_test_dataset_list = []
    pseudo_targets_per_session = len(pseudo_novel_cls) // pseudo_continual_session_num
    for s in range(test_continual_session_num):
        offline_session_targets = pseudo_novel_cls[0: s * pseudo_targets_per_session + pseudo_targets_per_session]
        offline_session_novel_samples = [offline_train_dataset_samples[target] for target in offline_session_targets]
        offline_session_novel_samples = [subsample_dataset(deepcopy(samples),  # random sample
                                                          np.random.choice(
                                                              np.array(list(range(len(samples.targets)))),
                                                              pseudo_each_session_novel_sample_num, replace=False))
                                        for samples in offline_session_novel_samples]
        offline_session_novel_dataset = subDataset_wholeDataset(offline_session_novel_samples)
        offline_pseudo_novel_unlabelled_list.append(offline_session_novel_dataset)
        offline_session_test_dataset = subDataset_wholeDataset([offline_test_dataset_samples[cls] for cls in (pseudo_old_cls + offline_session_targets)])
        offline_session_test_dataset.transform = test_transform
        offline_test_dataset_list.append(offline_session_test_dataset)

    # ------------------------------online old classes unlabeled samples------------------------------------------------
    online_old_unlabelled_indices_shuffle = np.array(list(range(len(online_old_unlabeled_samples[0]))))
    np.random.shuffle(online_old_unlabelled_indices_shuffle)
    online_old_unlabelled_indices_list = np.array_split(online_old_unlabelled_indices_shuffle,
                                                         test_continual_session_num)

    online_old_dataset_unlabelled_list = [subDataset_wholeDataset([subsample_dataset(deepcopy(sample), indices)
                                          for sample in online_old_unlabeled_samples]) for indices in
                                          online_old_unlabelled_indices_list]

    # ------------------------------online novel classes unlabeled samples----------------------------------------------
    novel_unlabelled_indices = set(whole_training_set.uq_idxs) - set(old_dataset_all.uq_idxs)
    novel_dataset_unlabelled = subsample_dataset(deepcopy(whole_training_set),
                                                 np.array(list(novel_unlabelled_indices)))
    # Get test set for all classes
    test_dataset = CustomCIFAR100(root=cifar_100_root, transform=test_transform, train=False)

    novel_targets_shuffle = np.array(list(set(np.array(novel_dataset_unlabelled.targets).tolist())))
    np.random.shuffle(novel_targets_shuffle)

    online_novel_dataset_unlabelled_list = []
    online_test_dataset_list = []
    targets_per_session = len(novel_targets_shuffle) // test_continual_session_num
    for s in range(test_continual_session_num):
        online_session_targets = novel_targets_shuffle[0: s * targets_per_session + targets_per_session]
        online_session_novel_samples = [subsample_classes(deepcopy(novel_dataset_unlabelled), include_classes=[targets])
                                        for targets in online_session_targets]
        online_session_novel_samples = [subsample_dataset(deepcopy(samples),  # random sample
                                                       np.random.choice(
                                                           np.array(list(range(len(samples.targets)))),
                                                           test_each_session_novel_sample_num, replace=False))
                                                                for samples in online_session_novel_samples]

        online_session_novel_dataset = subDataset_wholeDataset(online_session_novel_samples)
        online_novel_dataset_unlabelled_list.append(online_session_novel_dataset)
        online_session_test_dataset = subsample_classes(
            deepcopy(test_dataset), include_classes=list(train_classes) + online_session_targets.tolist())
        online_test_dataset_list.append(online_session_test_dataset)

    all_datasets = {
        'pretrain_pseudo_old_cls_train_samples': pretrain_pseudo_old_cls_train_samples,
        'pretrain_pseudo_old_cls_test_samples':pretrain_pseudo_old_cls_test_samples,
        'offline_pseudo_old_unlabelled_list': offline_pseudo_old_unlabelled_list,
        'offline_pseudo_novel_unlabelled_list': offline_pseudo_novel_unlabelled_list,
        'offline_test_dataset_list': offline_test_dataset_list,
        'online_old_dataset_unlabelled_list': online_old_dataset_unlabelled_list,
        'online_novel_dataset_unlabelled_list': online_novel_dataset_unlabelled_list,
        'online_test_dataset_list': online_test_dataset_list,
    }

    return all_datasets

if __name__ == '__main__':

    x = get_cifar_10_datasets(None, None, split_train_val=False,
                               train_classes=range(7), prop_train_labels=0.8)


    print('Printing lens...')
    for k, v in x.items():
        if v is not None:
            print(f'{k}: {len(v)}')

    print('Printing labelled and unlabelled overlap...')
    print(set.intersection(set(x['train_labelled'].uq_idxs), set(x['train_unlabelled'].uq_idxs)))
    print('Printing total instances in train...')
    print(len(set(x['train_labelled'].uq_idxs)) + len(set(x['train_unlabelled'].uq_idxs)))

    print(f'Num Labelled Classes: {len(set(x["train_labelled"].targets))}')
    print(f'Num Unabelled Classes: {len(set(x["train_unlabelled"].targets))}')
    print(f'Len labelled set: {len(x["train_labelled"])}')
    print(f'Len unlabelled set: {len(x["train_unlabelled"])}')

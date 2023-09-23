import argparse
import os
import sys
sys.path.append('/home/yananwu/myCode/MetaGCD/')

from torch.utils.data import DataLoader
import numpy as np

import torch
from torch.optim import SGD, lr_scheduler
from models import vision_transformer as vits

from project_utils.general_utils import set_seed, str2bool, time_str, hierarchical_print, Logger

from data.augmentations import get_transform
from data.get_datasets import get_class_splits, ContrastiveLearningViewGenerator, get_datasets


from config import dino_pretrain_path, exp_root
import time
import copy

from models.utils import SupConLoss, info_nce_logits, test_kmeans, generate_pseudo_labels, meta_update_model

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
            description='cluster',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # ==== running mode ====
    parser.add_argument('--run_mode', type=str, default='MetaTrain')  # ['OffTrain', 'MetaTrain', 'MetaTest']

    # === data ===
    parser.add_argument('--dataset_name', type=str, default='cifar100', help='options: cifar10, cifar100, tiny_imagenet')
    parser.add_argument('--prop_train_labels', type=float, default=0.8)
    parser.add_argument('--use_ssb_splits', type=str2bool, default=True)
    parser.add_argument('--transform', type=str, default='imagenet', help='pytorch-cifar, imagenet')

    # === model ===
    parser.add_argument('--model_name', type=str, default='vit_dino', help='Format is {model_name}_{pretrain}')
    parser.add_argument('--eval_funcs', nargs='+', help='Which eval functions to use', default=['v1', 'v2'])
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--base_model', type=str, default='vit_dino')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--sup_con_weight', type=float, default=0.5)
    parser.add_argument('--soft_threshold', type=float, default=0.85)
    parser.add_argument('--n_views', default=2, type=int)
    parser.add_argument('--contrast_unlabel_only', type=str2bool, default=False)

    parser.add_argument('--netModel_ckpt', type=str, default='/home/yananwu/myCode/MetaGCD/checkpoints/OffTrain_model.pth')
    parser.add_argument('--netHead_ckpt', type=str, default='/home/yananwu/myCode/MetaGCD/checkpoints/OffTrain_projection_head.pth')

    # === meta train ===
    parser.add_argument('--epochs_task', type=int, default=20, help='number of total tasks')
    # session 0 pretrain--------
    parser.add_argument('--pretrain_bs', type=int, default=512)
    parser.add_argument('--pretrain_lr', type=float, default=0.1)
    parser.add_argument('--pretrain_steps', default=50, type=int)
    # session 1 ~ n ---- (n is defined in dataset class)
    parser.add_argument('--metaTrain_bs', type=int, default=128)
    # inner loop
    parser.add_argument('--inner_lr', type=float, default=0.0001)
    parser.add_argument('--inner_steps', default=10, type=int)
    # outer loop
    parser.add_argument('--outer_lr', type=float, default=0.00001)
    parser.add_argument('--outer_steps', default=5, type=int)

    # === meta test ===
    parser.add_argument('--metaTest_bs', default=128, type=int)
    parser.add_argument('--test_lr', type=float, default=0.0001)
    parser.add_argument('--test_steps', default=20, type=int)
    parser.add_argument('--generate_pseudo_label', action='store_true', default=True)

    # === other ===
    # parser.add_argument('--exp_root', type=str, default=exp_root)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--grad_from_block', type=int, default=11)
    parser.add_argument('--log_interval', default=10, type=int, metavar='s', help='interval to output training logs')
    parser.add_argument('--desc', default='open_source_test', type=str, help='description for this running process')
    # ------------------------------------------------- init -----------------------------------------------------
    args = parser.parse_args()
    device = torch.device('cuda:0')
    set_seed(args.seed)

    # === save dir ===
    save_dir = os.path.join(exp_root, args.dataset_name, f'Time-{time_str()}')

    if args.desc is not None:
        save_dir += f'_{args.desc}'
    ckpt_save_dir = os.path.join(save_dir, 'model_ckpt')
    os.makedirs(ckpt_save_dir)
    # tb_save_dir = os.path.join(save_dir, 'tensorboard')
    # os.makedirs(tb_save_dir)

    log_path = os.path.join(save_dir, 'log.txt')
    sys.stdout = Logger(log_path, flush=True)
    # writer = SummaryWriter(log_dir=tb_save_dir)
    #=======

    args = get_class_splits(args)

    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)

    # === print args ===
    line_length = 120
    print('=' * line_length)
    print(args)
    print('=' * line_length)

    # ------------------------------------------------- model -----------------------------------------------------
    if args.base_model == 'vit_dino':

        # args.interpolation = 3
        args.crop_pct = 0.875
        pretrain_path = dino_pretrain_path

        # === NOTE: Hardcoded image size as we do not finetune the entire ViT model ===
        args.image_size = 224
        args.feat_dim = 768
        args.num_mlp_layers = 3
        args.mlp_out_dim = 65536

        model = vits.__dict__['vit_base']()

        if args.netModel_ckpt != '':
            model.load_state_dict(torch.load(args.netModel_ckpt))
            print(f'[Info] Loading ckpt file - {args.netModel_ckpt}')
        else:
            model.load_state_dict(torch.load(pretrain_path, map_location='cpu'))

        # === how much of base model to finetune ===
        for m in model.parameters():
            m.requires_grad = False

        # === Only finetune layers from block 'args.grad_from_block' onwards ===
        for name, m in model.named_parameters():
            if 'block' in name:
                block_num = int(name.split('.')[1])
                if block_num >= args.grad_from_block:
                    m.requires_grad = True
        model.to(device)
    else:
        raise NotImplementedError

    # === projection head ===
    projection_head = vits.__dict__['DINOHead'](in_dim=args.feat_dim,
                                                out_dim=args.mlp_out_dim, nlayers=args.num_mlp_layers)
    if args.netHead_ckpt != '':
        projection_head.load_state_dict(torch.load(args.netHead_ckpt))
        print(f'[Info] Loading ckpt file - {args.netHead_ckpt}')

    projection_head.to(device)

    # ---------------------------------------------------- data --------------------------------------------------------
    # === CONTRASTIVE TRANSFORM ===
    train_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
    train_transform = ContrastiveLearningViewGenerator(base_transform=train_transform, n_views=args.n_views)


    def train_one_epoch(projection_head, model, train_loader, optimizer, generate_soft_label=True):
        projection_head.train()
        model.train()
        total_iter_num = len(train_loader)
        for iter_idx, batch in enumerate(train_loader):

            images, class_labels, uq_idxs, mask_lab = batch
            mask_lab = mask_lab[:, 0]

            class_labels, mask_lab = class_labels.to(device), mask_lab.to(device).bool()
            images = torch.cat(images, dim=0).to(device)

            # Extract features with base model
            features_base = model(images)

            # Pass features through projection head
            features = projection_head(features_base)

            # L2-normalize features
            features = torch.nn.functional.normalize(features, dim=-1)

            # Choose which instances to run the contrastive loss on
            if args.contrast_unlabel_only:
                # Contrastive loss only on unlabelled instances
                f1, f2 = [f[~mask_lab] for f in features.chunk(2)]
                con_feats = torch.cat([f1, f2], dim=0)
            else:
                # Contrastive loss for all examples
                con_feats = features

            contrastive_logits, contrastive_labels = info_nce_logits(features=con_feats, args=args, device=device)
            contrastive_loss = torch.nn.CrossEntropyLoss()(contrastive_logits, contrastive_labels)

            if generate_soft_label:
                simi_mask, pseudo_mask, w_attention = generate_pseudo_labels(features, k_threshold=args.soft_threshold)
                f1, f2 = [f[pseudo_mask] for f in features.chunk(2)]
                sup_con_feats = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

                if len(sup_con_feats) == 0:
                    sup_con_loss = torch.tensor(0)
                    loss = contrastive_loss
                else:
                    sup_con_loss = sup_con_crit(sup_con_feats, mask=simi_mask, weight=w_attention, device=device)
                    loss = (1 - args.sup_con_weight) * contrastive_loss + args.sup_con_weight * sup_con_loss
            else:
                f1, f2 = [f[mask_lab] for f in features.chunk(2)]
                sup_con_feats = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                sup_con_labels = class_labels[mask_lab]
                sup_con_loss = sup_con_crit(sup_con_feats, labels=sup_con_labels, device=device)
                loss = (1 - args.sup_con_weight) * contrastive_loss + args.sup_con_weight * sup_con_loss

            # Train acc
            _, pred = contrastive_logits.max(1)
            acc = (pred == contrastive_labels).float().mean().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (iter_idx + 1) % args.log_interval == 0:
                hierarchical_print(f'Iter [{iter_idx + 1}/{total_iter_num}] '
                                   f'Loss_U:{contrastive_loss.item():.4f} '
                                   f'Loss_S:{sup_con_loss.item():.4f} '
                                   f'Loss_AVG:{loss.item():.4f} '
                                   f'Train_acc:{acc:.4f} ',
                                   length=line_length, mode='in', level=2)

    # ------------------------------------------------- meta train -----------------------------------------------------
    if 'MetaTrain' in args.run_mode:
        print('[Info] Starting process - MetaTrain')

        for epoch_task in range(args.epochs_task):
            hierarchical_print(f'Meta Task [{epoch_task + 1}/{args.epochs_task}]',
                               length=line_length, mode='out', level=0)

            if epoch_task > 0:
                projection_head.load_state_dict(torch.load(os.path.join(ckpt_save_dir,
                                        f'MetaTrain_T_{epoch_task-1}_MetaCL_model_projection_head.pth')))
                model.load_state_dict(torch.load(os.path.join(ckpt_save_dir, f'MetaTrain_T_{epoch_task-1}_MetaCL_model_model.pth')))

            # === prepare data ===
            hierarchical_print(f'[Info] Loading dataset', length=line_length, mode='out', level=1)
            pretrain_train_dataset, pretrain_test_dataset, offline_session_train_dataset_list, offline_session_test_dataset_list, \
                online_session_train_dataset_list, online_session_test_dataset_list, datasets = get_datasets(
                args.dataset_name, train_transform, test_transform, args)
            hierarchical_print(f'[Info] dataset final', length=line_length, mode='out', level=1)

            args.train_classes = list(np.unique(pretrain_train_dataset.old_unlabelled_dataset.targets))
            args.num_labeled_classes = len(args.train_classes)

            # === define loss ===
            sup_con_crit = SupConLoss()

            # session 0 ------------------------------------------------------------------------------------------------
            hierarchical_print(f'Pseudo PreTrain - Session 0', length=line_length, mode='out', level=1)

            # === prepare data ===
            pretrain_train_loader = DataLoader(pretrain_train_dataset, num_workers=args.num_workers,
                                               batch_size=args.pretrain_bs, shuffle=True, drop_last=True)
            pretrain_test_loader = DataLoader(pretrain_test_dataset, num_workers=args.num_workers,
                                              batch_size=args.pretrain_bs,
                                              shuffle=True)

            # ==== define optimizer ====
            pretrain_optimizer = SGD(list(projection_head.parameters()) + list(model.parameters()), lr=args.pretrain_lr,
                                     momentum=args.momentum,
                                     weight_decay=args.weight_decay)

            # ==== start training ====
            # pre_best_test_acc, pre_best_epoch = 0, -1
            # for pretrain_step in range(args.pretrain_steps):
            #     hierarchical_print(f'Epoch [{pretrain_step + 1}/{args.pretrain_steps}]',
            #                        length=line_length, mode='out', level=2)
            #     train_one_epoch(projection_head, model, pretrain_train_loader, pretrain_optimizer, generate_soft_label=False)
            #
            #     # === calculate test dataset acc ===
            #     s_time = time.time()
            #     hierarchical_print(f'Evalution', length=line_length, mode='out', level=2)
            #
            #     with torch.no_grad():
            #         args.num_cur_unlabeled_classes = 0
            #         old_acc_test = test_kmeans(model, pretrain_test_loader, stage='pretrain', epoch=pretrain_step,
            #                                    save_name='Test ACC', args=args, device=device)
            #
            #     if old_acc_test > pre_best_test_acc:
            #         pre_best_test_acc = old_acc_test
            #         pre_best_epoch = pretrain_step
            #         torch.save(model.state_dict(), os.path.join(ckpt_save_dir, 'Pretrain_model.pth'))
            #         torch.save(projection_head.state_dict(),
            #                    os.path.join(ckpt_save_dir, 'Pretrain__projection_head.pth'))
            #
            #     hierarchical_print(f'Epoch[{pretrain_step + 1}/{args.pretrain_steps}]: '
            #                        f'old_acc_test: {old_acc_test:.4f} ',
            #                        length=line_length, mode='in', level=3)
            #     hierarchical_print(f'Total Time: {time.time() - s_time:.2f}s',
            #                            length=line_length, mode='in', level=3)
            # hierarchical_print(f'Meta Task [{epoch_task + 1}/{args.epochs_task}] - PreTrain Best Results',
            #                    length=line_length, mode='out', level=2)
            # hierarchical_print(f'Epoch[{pre_best_epoch + 1}/{args.pretrain_steps}]: '
            #                    f'old_cls_acc: {pre_best_test_acc:.4f} ',
            #                    length=line_length, mode='in', level=3)
            # hierarchical_print('', length=line_length, mode='out', level=2)

            # # session 1-n ----------------------------------------------------------------------------------------------
            # ==== define optimizer ====
            meta_train_in_optimizer = SGD(list(projection_head.parameters()) + list(model.parameters()),
                                          lr=args.inner_lr, momentum=args.momentum, weight_decay=args.weight_decay)

            continual_session_num = len(offline_session_train_dataset_list)
            step_best_results_mode2 = {'session_list': list(range(continual_session_num)),
                                       'old_list': [0] * continual_session_num,
                                       'novel_list': [0] * continual_session_num,
                                       'all_list': [0] * continual_session_num,
                                       'epoch_list': [-1] * continual_session_num}

            for session in range(continual_session_num):
                hierarchical_print(f'Pseudo Continual Learning - '
                                   f'Session [{session + 1}/{continual_session_num}]',
                                   length=line_length, mode='out', level=1)

                # === prepare data ===
                offline_session_train_loader = DataLoader(offline_session_train_dataset_list[session],
                                                  num_workers=args.num_workers,
                                                  batch_size=args.metaTrain_bs, shuffle=True, drop_last=True)
                offline_session_test_loader = DataLoader(offline_session_test_dataset_list[session],
                                                 num_workers=args.num_workers,
                                                 batch_size=args.metaTrain_bs,
                                                 shuffle=True)

                # ==== save outer model params ====
                projection_head_outer_state_dict_copy = copy.deepcopy(projection_head.state_dict())
                model_outer_state_dict_copy = copy.deepcopy(model.state_dict())

                # ==== inner loop ====
                for inner_step in range(args.inner_steps):
                    hierarchical_print(f'Meta - Inner Loop [{inner_step + 1}/{args.inner_steps}]',
                                       length=line_length, mode='out', level=2)
                    train_one_epoch(projection_head, model, offline_session_train_loader, meta_train_in_optimizer, generate_soft_label=True)

                    # === calculate test dataset acc ===
                    s_time = time.time()
                    hierarchical_print(f'Evalution', length=line_length, mode='out', level=2)

                    with torch.no_grad():
                        args.num_cur_unlabeled_classes = len(np.unique(offline_session_train_dataset_list[session].novel_unlabelled_dataset.targets))
                        all_acc_test, old_acc_test, new_acc_test = test_kmeans(model, offline_session_test_loader,
                                                       stage='metaTest', epoch=inner_step, save_name='Test ACC', args=args, device=device)
                    hierarchical_print(f'Epoch[{inner_step + 1}/{args.inner_steps}]: '
                                       f'old_acc: {old_acc_test:.4f} '
                                       f'novel_acc: {new_acc_test:.4f} '
                                       f'all_acc: {all_acc_test:.4f}',
                                       length=line_length, mode='in', level=3)
                    hierarchical_print(f'Total Time: {time.time() - s_time:.2f}s',
                                       length=line_length, mode='in', level=3)
                hierarchical_print('', length=line_length, mode='out', level=2)

                # ==== outer loop ====
                meta_train_out_optimizer = SGD(list(projection_head.parameters()) + list(model.parameters()),
                                               lr=args.outer_lr, momentum=args.momentum,
                                               weight_decay=args.weight_decay)

                for outer_step in range(args.outer_steps):
                    hierarchical_print(f'Meta - Outer Loop [{outer_step + 1}/{args.outer_steps}]',
                                       length=line_length, mode='out', level=2)

                    validation_loss_store = 0
                    validation_loss = 0
                    dummy_input_image, dummy_input_labels, dummy_input_mask = [], [], []

                    projection_head.train()
                    model.train()
                    total_iter_num = len(offline_session_train_loader)
                    for iter_idx, batch in enumerate(offline_session_train_loader):
                        images, class_labels, uq_idxs, mask_lab = batch
                        mask_lab = mask_lab[:, 0] + 1

                        class_labels, mask_lab = class_labels.to(device), mask_lab.to(device).bool()
                        images = torch.cat(images, dim=0).to(device)

                        if iter_idx == 0:
                            dummy_input_image = images
                            dummy_input_labels = class_labels
                            dummy_input_mask = mask_lab

                        features = model(images)
                        features = projection_head(features)
                        features = torch.nn.functional.normalize(features, dim=-1)

                        contrastive_logits, contrastive_labels = info_nce_logits(features=features, args=args,
                                                                                 device=device)
                        contrastive_loss = torch.nn.CrossEntropyLoss()(contrastive_logits, contrastive_labels)

                        # Supervised contrastive loss
                        f1, f2 = [f[mask_lab] for f in features.chunk(2)]
                        sup_con_feats = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                        sup_con_labels = class_labels[mask_lab]

                        if len(sup_con_feats) == 0:
                            sup_con_loss = torch.tensor(0)
                            loss = contrastive_loss
                        else:
                            sup_con_loss = sup_con_crit(sup_con_feats, labels=sup_con_labels, device=device)
                            # Total loss
                            loss = (1 - args.sup_con_weight) * contrastive_loss + args.sup_con_weight * sup_con_loss

                        validation_loss_store = validation_loss_store + loss.item()

                        # Train acc
                        _, pred = contrastive_logits.max(1)  # 256
                        acc = (pred == contrastive_labels).float().mean().item()

                        if (iter_idx + 1) == total_iter_num:
                            # Store the loss
                            validation_loss = loss
                            validation_loss.data = torch.FloatTensor(
                                [validation_loss_store / total_iter_num]).cuda()

                        if (iter_idx + 1) % args.log_interval == 0:
                            hierarchical_print(f'Iter [{iter_idx + 1}/{total_iter_num}] '
                                               f'Loss_U:{contrastive_loss.item():.4f} '
                                               f'Loss_S:{sup_con_loss.item():.4f} '
                                               f'Loss_AVG:{loss.item():.4f} '
                                               f'Train_acc:{acc:.4f} ',
                                               length=line_length, mode='in', level=3)

                    # Compute Validation Grad
                    projection_head.load_state_dict(projection_head_outer_state_dict_copy)
                    model.load_state_dict(model_outer_state_dict_copy)

                    torch.autograd.set_detect_anomaly(True)
                    projection_head_trainable_weights = [p for n, p in projection_head.named_parameters() if
                                                         p.requires_grad]
                    model_trainable_weights = [p for n, p in model.named_parameters() if p.requires_grad]
                    grads = torch.autograd.grad(validation_loss,
                                                projection_head_trainable_weights + model_trainable_weights,
                                                create_graph=True)
                    # grads = torch.autograd.grad(validation_loss, list(projection_head.parameters()) + list(model.parameters()))

                    meta_grads = {name: g for ((name, _), g) in zip(
                        projection_head.named_parameters() + model.named_parameters(), grads)}

                    # ==== Meta Update ====
                    # Dummy Forward Pass
                    dummy_features = model(dummy_input_image)
                    dummy_features = projection_head(dummy_features)
                    dummy_features = torch.nn.functional.normalize(dummy_features, dim=-1)

                    contrastive_logits, contrastive_labels = info_nce_logits(features=dummy_features, args=args,
                                                                             device=device)
                    contrastive_loss = torch.nn.CrossEntropyLoss()(contrastive_logits, contrastive_labels)

                    # Supervised contrastive loss
                    f1, f2 = [f[dummy_input_mask] for f in dummy_features.chunk(2)]
                    sup_con_feats = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                    sup_con_labels = dummy_input_labels[dummy_input_mask]

                    sup_con_loss = sup_con_crit(sup_con_feats, labels=sup_con_labels)
                    loss = (1 - args.sup_con_weight) * contrastive_loss + args.sup_con_weight * sup_con_loss

                    meta_update_model(projection_head, model, meta_train_out_optimizer, loss, meta_grads)

                    projection_head_outer_state_dict_copy = copy.deepcopy(projection_head.state_dict())
                    model_outer_state_dict_copy = copy.deepcopy(model.state_dict())

                    # === calculate acc ===
                    s_time = time.time()
                    hierarchical_print(f'Evalution',
                                       length=line_length, mode='out', level=2)
                    cur_step_results = {'session_list': [], 'old_list': [], 'novel_list': [], 'all_list': [],
                                        'old_avg': 0, 'novel_avg': 0, 'all_avg': 0, 'epoch': -1}

                    with torch.no_grad():
                        args.num_unlabeled_classes = len(
                            np.unique(offline_session_train_dataset_list[session].novel_unlabelled_dataset.targets))
                        all_acc_test, old_acc_test, new_acc_test = test_kmeans(model, offline_session_test_loader,
                                                                               stage='metaTest', epoch=outer_step,
                                                                               save_name='Test ACC', args=args)

                    old_acc_test = torch.tensor(old_acc_test)
                    new_acc_test = torch.tensor(new_acc_test)
                    all_acc_test = torch.tensor(all_acc_test)

                    cur_step_results['session_list'].append(session + 1)
                    cur_step_results['old_list'].append(old_acc_test)
                    cur_step_results['novel_list'].append(new_acc_test)
                    cur_step_results['all_list'].append(all_acc_test)

                    if new_acc_test > step_best_results_mode2['novel_list'][session]:
                        step_best_results_mode2['old_list'][session] = old_acc_test
                        step_best_results_mode2['novel_list'][session] = new_acc_test
                        step_best_results_mode2['all_list'][session] = all_acc_test
                        step_best_results_mode2['epoch_list'][session] = outer_step

                    hierarchical_print(f'Session[{session + 1}/{continual_session_num}]-'
                                       f'Epoch[{outer_step + 1}/{args.outer_steps}]: '
                                       f'old_acc: {old_acc_test:.4f} '
                                       f'novel_acc: {new_acc_test:.4f} '
                                       f'all_acc: {all_acc_test:.4f}',
                                       length=line_length, mode='in', level=3)

                    cur_step_results['old_avg'] = torch.mean(torch.stack(cur_step_results['old_list']))
                    cur_step_results['novel_avg'] = torch.mean(torch.stack(cur_step_results['novel_list']))
                    cur_step_results['all_avg'] = torch.mean(torch.stack(cur_step_results['all_list']))
                    cur_step_results['epoch'] = outer_step

                    hierarchical_print(f'Total Time: {time.time() - s_time:.2f}s',
                                       length=line_length, mode='in', level=3)

                    hierarchical_print('', length=line_length, mode='out', level=2)

                hierarchical_print(f'Session[{session + 1}] Best Results - Mode 2', length=line_length, mode='out',
                                   level=2)
                for s in range(session + 1):
                    hierarchical_print(f'Session[{s + 1}/{session + 1}]-'
                                       f'Epoch[{step_best_results_mode2["epoch_list"][s] + 1}/{args.outer_steps}]: '
                                       f'old: {step_best_results_mode2["old_list"][s]:.4f} '
                                       f'novel: {step_best_results_mode2["novel_list"][s]:.4f} '
                                       f'all: {step_best_results_mode2["all_list"][s]:.4f}',
                                       length=line_length, mode='in', level=3)
                hierarchical_print(f'Session[1-{session + 1}] AVG -'
                                   f'Epoch[{[e + 1 for e in step_best_results_mode2["epoch_list"][:(s + 1)]]}/{args.outer_steps}]: '
                                   f'old: {torch.mean(torch.stack(step_best_results_mode2["old_list"][:(s + 1)])):.4f} '
                                   f'novel: {torch.mean(torch.stack(step_best_results_mode2["novel_list"][:(s + 1)])):.4f} '
                                   f'all: {torch.mean(torch.stack(step_best_results_mode2["all_list"][:(s + 1)])):.4f}',
                                   length=line_length, mode='in', level=3)
                hierarchical_print('', length=line_length, mode='out', level=2)

            hierarchical_print('', length=line_length, mode='out', level=1)

            torch.save(projection_head.state_dict(),
                       os.path.join(ckpt_save_dir,
                                    f'MetaTrain_T_{epoch_task}_MetaCL_model_projection_head.pth'))
            torch.save(model.state_dict(),
                       os.path.join(ckpt_save_dir, f'MetaTrain_T_{epoch_task}_MetaCL_model_model.pth'))

 # -------------------------------------------------- meta test -----------------------------------------------------
    if 'MetaTest' in args.run_mode:
        print('=' * line_length)
        print('[Info] Starting process - MetaTest')

        # ==== prepare data ====
        print(f'[Info] Loading dataset')
        offline_train_dataset, offline_test_dataset, \
        online_session_train_dataset_list, \
        online_session_test_dataset_list, datasets = get_datasets(args.dataset_name, train_transform, test_transform,
                                                                   args)
        print(f'[Info] dataset final')

        # === define loss ===
        sup_con_crit = SupConLoss()

        # best_test_acc_lab = 0
        continual_session_num = len(online_session_train_dataset_list)

        # ==== continual session ====
        step_best_results_mode2 = {'session_list': list(range(continual_session_num)),
                                   'old_list': [0] * continual_session_num,
                                   'novel_list': [0] * continual_session_num,
                                   'all_list': [0] * continual_session_num,
                                   'epoch_list': [-1] * continual_session_num}

        for session in range(continual_session_num):
            hierarchical_print(f'Session [{session + 1}/{continual_session_num}]',
                               length=line_length, mode='out', level=0)

            args.num_novel_class_per_session = len(np.unique(online_session_train_dataset_list[session].novel_unlabelled_dataset.targets))
            args.num_pre_classes = args.num_labeled_classes + args.num_novel_class_per_session * session

            # === define optimizer ===
            meta_test_optimizer = SGD(list(projection_head.parameters()) + list(model.parameters()), lr=args.test_lr,
                                      momentum=args.momentum,
                                      weight_decay=args.weight_decay)

            # meta_test_lr_scheduler = lr_scheduler.MultiStepLR(meta_test_optimizer, milestones=[5], gamma=0.1)

            # prepare each session data
            meta_test_train_dataset = online_session_train_dataset_list[session]
            meta_test_test_dataset = online_session_test_dataset_list[session]

            meta_test_train_loader = DataLoader(meta_test_train_dataset, num_workers=args.num_workers,
                                              batch_size=args.metaTest_bs, shuffle=True, drop_last=True)
            meta_test_test_loader = DataLoader(meta_test_test_dataset, num_workers=args.num_workers,
                                             batch_size=args.metaTest_bs, shuffle=False)

            # === start train ===
            for step in range(args.test_steps):
                hierarchical_print(f'Session [{session + 1}/{continual_session_num}] - '
                                   f'Epoch [{step + 1}/{args.test_steps}] - ',
                                   # f'lr {meta_test_lr_scheduler.get_last_lr()}',
                                   length=line_length, mode='out', level=1)

                train_one_epoch(projection_head, model, meta_test_train_loader, meta_test_optimizer, generate_soft_label=True)

                # meta_test_lr_scheduler.step()

                # === calculate acc ===
                s_time = time.time()
                hierarchical_print(f'Evalution', length=line_length, mode='out', level=1)
                cur_step_results = {'session_list': [], 'old_list': [], 'novel_list': [], 'all_list': [],
                                    'old_avg': 0, 'novel_avg': 0, 'all_avg': 0, 'epoch': -1}

                with torch.no_grad():
                    args.num_cur_unlabeled_classes = len(
                        np.unique(online_session_train_dataset_list[session].novel_unlabelled_dataset.targets))
                    all_acc_test, old_acc_test, new_acc_test = test_kmeans(model, meta_test_test_loader, stage='metaTest',
                                                                           epoch=step, save_name='Test ACC',
                                                                           args=args, device=device)

                old_acc_test = torch.tensor(old_acc_test)
                new_acc_test = torch.tensor(new_acc_test)
                all_acc_test = torch.tensor(all_acc_test)

                cur_step_results['session_list'].append(session + 1)
                cur_step_results['old_list'].append(old_acc_test)
                cur_step_results['novel_list'].append(new_acc_test)
                cur_step_results['all_list'].append(all_acc_test)

                if new_acc_test > step_best_results_mode2['novel_list'][session]:
                    step_best_results_mode2['old_list'][session] = old_acc_test
                    step_best_results_mode2['novel_list'][session] = new_acc_test
                    step_best_results_mode2['all_list'][session] = all_acc_test
                    step_best_results_mode2['epoch_list'][session] = step

                hierarchical_print(f'Session[{session + 1}/{continual_session_num}]-'
                                   f'Epoch[{step + 1}/{args.test_steps}]: '
                                   f'old_acc: {old_acc_test:.4f} '
                                   f'novel_acc: {new_acc_test:.4f} '
                                   f'all_acc: {all_acc_test:.4f}',
                                   length=line_length, mode='in', level=2)

                cur_step_results['old_avg'] = torch.mean(torch.stack(cur_step_results['old_list']))
                cur_step_results['novel_avg'] = torch.mean(torch.stack(cur_step_results['novel_list']))
                cur_step_results['all_avg'] = torch.mean(torch.stack(cur_step_results['all_list']))
                cur_step_results['epoch'] = step

                hierarchical_print(f'Total Time: {time.time() - s_time:.2f}s',
                                   length=line_length, mode='in', level=2)

            hierarchical_print(f'Session[{session + 1}] Best Results - Mode 2', length=line_length, mode='out', level=1)
            for s in range(session + 1):
                hierarchical_print(f'Session[{s + 1}/{session + 1}]-'
                                   f'Epoch[{step_best_results_mode2["epoch_list"][s] + 1}/{args.test_steps}]: '
                                   f'old: {step_best_results_mode2["old_list"][s]:.4f} '
                                   f'novel: {step_best_results_mode2["novel_list"][s]:.4f} '
                                   f'all: {step_best_results_mode2["all_list"][s]:.4f}',
                                   length=line_length, mode='in', level=2)
            hierarchical_print(f'Session[1-{session + 1}] AVG -'
                               f'Epoch[{[e + 1 for e in step_best_results_mode2["epoch_list"][:(s + 1)]]}/{args.test_steps}]: '
                               f'old: {torch.mean(torch.stack(step_best_results_mode2["old_list"][:(s + 1)])):.4f} '
                               f'novel: {torch.mean(torch.stack(step_best_results_mode2["novel_list"][:(s + 1)])):.4f} '
                               f'all: {torch.mean(torch.stack(step_best_results_mode2["all_list"][:(s + 1)])):.4f}',
                               length=line_length, mode='in', level=2)
            hierarchical_print('', length=line_length, mode='out', level=1)
        hierarchical_print('', length=line_length, mode='out', level=0)

        torch.save(model.state_dict(), os.path.join(ckpt_save_dir, 'MetaTest_model.pth'))
        torch.save(projection_head.state_dict(), os.path.join(ckpt_save_dir, 'MetaTest_projection_head.pth'))
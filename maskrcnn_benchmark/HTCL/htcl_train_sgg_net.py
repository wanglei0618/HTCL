# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Basic training script for PyTorch
"""

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip
import numpy as np
import argparse
import os
import time
import datetime
import sys

import torch
from torch.nn.utils import clip_grad_norm_
from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.solver import make_lr_scheduler
from maskrcnn_benchmark.solver import make_optimizer
from maskrcnn_benchmark.engine.trainer import reduce_loss_dict
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.checkpoint import clip_grad_norm
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank, all_gather
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.logger import setup_logger, debug_print
from maskrcnn_benchmark.utils.miscellaneous import mkdir, save_config
from maskrcnn_benchmark.utils.metric_logger import MetricLogger
from maskrcnn_benchmark.utils.model_serialization import load_state_dict
from maskrcnn_benchmark.modeling.utils import cat

# See if we can use apex.DistributedDataParallel instead of the torch default,
# and enable mixed-precision via apex.amp
# try:
#     from apex import amp
# except ImportError:
#     raise ImportError('Use APEX for multi-precision via apex.amp')


def htcl_train_sgg_net(cfg, local_rank, distributed, logger):
    cfg.MODEL.distributed = distributed
    debug_print(logger, 'prepare training')
    model = build_detection_model(cfg)
    debug_print(logger, 'end model construction')

    # modules that should be always set in eval mode
    # their eval() method should be called after model.train() is called
    eval_modules = (model.rpn, model.backbone, model.roi_heads.box,)
    fix_eval_modules(eval_modules)

    for key, value in model.named_parameters():
        if value.requires_grad:
            print(key)

    # NOTE, we slow down the LR of the layers start with the names in slow_heads
    if cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR == "IMPPredictor":
        slow_heads = ["roi_heads.relation.box_feature_extractor",
                      "roi_heads.relation.union_feature_extractor.feature_extractor", ]
    else:
        slow_heads = []

    # load pretrain layers to new layers
    load_mapping = {"roi_heads.relation.box_feature_extractor": "roi_heads.box.feature_extractor",
                    "roi_heads.relation.union_feature_extractor.feature_extractor": "roi_heads.box.feature_extractor"}

    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    num_batch = cfg.SOLVER.IMS_PER_BATCH

    optimizer = make_optimizer(cfg, model, logger, slow_heads=slow_heads, slow_ratio=10.0,
                               rl_factor=float(num_batch))
    scheduler = make_lr_scheduler(cfg, optimizer, logger)
    debug_print(logger, 'end optimizer and shcedule')
    # Initialize mixed-precision training
    # use_mixed_precision = cfg.DTYPE == "float16"
    # amp_opt_level = 'O1' if use_mixed_precision else 'O0'
    # model, optimizer = amp.initialize(model, optimizer, opt_level=amp_opt_level)
    save_to_disk = get_rank() == 0
    output_dir = cfg.OUTPUT_DIR

    debug_print(logger, 'end distributed')
    arguments = {}
    arguments["iteration"] = 0
    arguments["last_val_results"] = 0
    arguments["best_val_results"] = 0

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
            find_unused_parameters=True,
        )

    checkpointer = DetectronCheckpointer(
        cfg, model, optimizer, scheduler, output_dir, save_to_disk, custom_scheduler=True
    )

    # if there is certain checkpoint in output_dir, load it, else load pretrained detector
    if checkpointer.has_checkpoint():
        extra_checkpoint_data = checkpointer.load(cfg.MODEL.PRETRAINED_DETECTOR_CKPT,
                                                  update_schedule=True)
        arguments.update(extra_checkpoint_data)
    else:
        # load_mapping is only used when we init current model from detection model.
        checkpointer.load(cfg.MODEL.PRETRAINED_DETECTOR_CKPT, with_optim=False, load_mapping=load_mapping)

    if cfg.MODEL.ct_loss:
        ct_loss = CenterLoss(cfg.MODEL.ROI_RELATION_HEAD.NUM_CLASSES, cfg.MODEL.ROI_RELATION_HEAD.LAST_FEATS_DIM)
        center_optimizer = torch.optim.SGD(ct_loss.parameters(), lr=0.5)
        center_weight = 0.005
        # ct_loss, center_optimizer = amp.initialize(ct_loss, center_optimizer, opt_level=amp_opt_level, verbosity=0)
        if distributed:
            ct_loss = torch.nn.parallel.DistributedDataParallel(
                ct_loss, device_ids=[local_rank], output_device=local_rank,
                # this should be removed if we update BatchNorm stats
                broadcast_buffers=False,
                find_unused_parameters=True)
        ct_dir = output_dir + '/ct'
        mkdir(ct_dir)
        ct_checkpointer = DetectronCheckpointer(cfg, ct_loss, center_optimizer, save_dir=ct_dir,
                                                save_to_disk=save_to_disk, logger=logger)
        if ct_checkpointer.has_checkpoint():
            extra_checkpoint_data_ct = ct_checkpointer.load()
            arguments.update(extra_checkpoint_data_ct)

    debug_print(logger, 'end load checkpointer')

    train_data_loader = make_data_loader(
        cfg,
        mode='train',
        is_distributed=distributed,
        start_iter=arguments["iteration"],
    )

    val_data_loaders = make_data_loader(
        cfg,
        mode='val',
        is_distributed=distributed,
    )
    debug_print(logger, 'end dataloader')

    if cfg.SOLVER.PRE_VAL:
        logger.info("Validate before training")
        run_val(cfg, model, val_data_loaders, distributed, logger)

    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(train_data_loader)
    start_iter = arguments["iteration"]
    start_training_time = time.time()
    end = time.time()

    print_first_grad = True
    for iteration, (images, targets, _) in enumerate(train_data_loader, start_iter):
        if any(len(target) < 1 for target in targets):
            logger.error(
                f"Iteration={iteration + 1} || Image Ids used for training {_} || targets Length={[len(target) for target in targets]}")
        data_time = time.time() - end
        iteration = iteration + 1

        arguments["iteration"] = iteration

        model.train()
        images = images.to(device)
        targets = [target.to(device) for target in targets]
        optimizer.zero_grad()
        if cfg.MODEL.ct_loss:
            center_optimizer.zero_grad()
            loss_dict, q_feats, rel_labels, cut_labels, relation_logits = model(images, targets)
            center_update, loss_ct = ct_loss(q_feats, cut_labels.long(), relation_logits, cat(rel_labels).long())
            loss_dict.update(dict(center_update=center_weight * center_update))
            loss_dict.update(dict(loss_ct=cfg.MODEL.w_ct * loss_ct))
        else:
            loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)

        # Note: If mixed precision is not used, this ends up doing nothing
        # Otherwise apply loss scaling for mixed-precision recipe
        # if cfg.MODEL.ct_loss:
        #     with amp.scale_loss(losses, [optimizer, center_optimizer]) as scaled_losses:
        #         scaled_losses.backward()
        # else:
        #     with amp.scale_loss(losses, optimizer) as scaled_losses:
        #         scaled_losses.backward()
        losses.backward()

        # add clip_grad_norm from MOTIFS, tracking gradient, used for debug
        verbose = (iteration % cfg.SOLVER.VAL_PERIOD) == 0 or print_first_grad  # print grad or not
        print_first_grad = False
        clip_grad_norm([(n, p) for n, p in model.named_parameters() if p.requires_grad],
                       max_norm=cfg.SOLVER.GRAD_NORM_CLIP, logger=logger, verbose=verbose, clip=True)
        optimizer.step()

        if cfg.MODEL.ct_loss:
            # multiple (1./alpha) in order to remove the effect of alpha on updating centers
            for param in ct_loss.parameters():
                param.grad.data *= (1. / (center_weight + 1e-12))
            center_optimizer.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)
        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % int(cfg.SOLVER.VAL_PERIOD/20) == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[-1]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )

        val_result = None  # used for scheduler updating
        if cfg.SOLVER.TO_VAL and iteration % cfg.SOLVER.VAL_PERIOD == 0:
            logger.info("Start validating")
            val_result = run_val(cfg, model, val_data_loaders, distributed, logger)
            arguments["last_val_results"] = val_result
            if arguments["last_val_results"] >= arguments["best_val_results"]:
                arguments["best_val_results"] = arguments["last_val_results"]
            logger.info("Validation Result: %.4f" % val_result)

            checkpointer.save("model_{:07d}".format(iteration), **arguments)
            if cfg.MODEL.ct_loss:
                ct_checkpointer.save("feats_center_{:07d}".format(iteration), **arguments)
        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)
            if cfg.MODEL.ct_loss:
                ct_checkpointer.save("feats_center_final", **arguments)

        # scheduler should be called after optimizer.step() in pytorch>=1.1.0
        # https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
        if cfg.SOLVER.SCHEDULE.TYPE == "WarmupReduceLROnPlateau":
            scheduler.step(val_result, epoch=iteration)
            if scheduler.stage_count >= cfg.SOLVER.SCHEDULE.MAX_DECAY_STEP:
                logger.info("Trigger MAX_DECAY_STEP at iteration {}.".format(iteration))
                break
        else:
            scheduler.step()

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )

    with open(output_dir + "/End_traning", 'w') as f:
        f.write('End sgg_net training')

    if not cfg.TEST.Skip_test:
        checkpointer.load()
        run_test(cfg, model, distributed, logger)


class CenterLoss(nn.Module):
    def __init__(self, num_classes=51, feat_dim=1024):
        super(CenterLoss, self).__init__()
        self.num_class = num_classes
        self.num_feature = feat_dim
        self.centers = nn.Parameter(torch.randn(self.num_class, self.num_feature).cuda())
        self.pred_fre = [0, 31, 20, 48, 30, 22, 29, 8, 50, 21, 1, 43, 49, 40, 23, 38, 41, 6,
                         7, 33, 11, 46, 16, 47, 25, 19, 5, 9, 35, 24, 10, 4, 14, 13, 12, 36,
                         44, 42, 32, 2, 26, 28, 45, 3, 17, 18, 34, 37, 27, 39, 15]

        self.head_pred_idx = torch.tensor(self.pred_fre[:cfg.MODEL.num_center_head_cls]).cuda()
        self.center_head_pred_idx = torch.tensor(self.pred_fre[:cfg.MODEL.num_center_head_cls]).cuda()
        self.body_tail = self.pred_fre[cfg.MODEL.num_center_head_cls:]
        self.LogSoftmax = nn.LogSoftmax(dim=1)
        self.criterion_loss = nn.CrossEntropyLoss()

    def head_ct_loss(self, x, labels):
        head_mask = torch.zeros(labels.shape[0], dtype=torch.long).cuda()
        for idx in self.center_head_pred_idx:
            head_mask[labels == idx] = 1
        center = self.centers[labels].clone().detach()
        dist = (x-center).pow(2).sum(dim=-1) * head_mask
        loss = torch.clamp(dist, min=1e-8, max=1e+8).mean(dim=-1)
        return loss

    def forward(self, x, cut_labels, logits, labels):
        x_ct_update = x[cut_labels != 0, :].clone().detach()
        cut_labels_fg = cut_labels[cut_labels != 0]
        center = self.centers[cut_labels_fg]
        dist = (x_ct_update-center).pow(2).sum(dim=-1)
        center_update = torch.clamp(dist, min=1e-8, max=1e+8).mean(dim=-1)
        head_ct_loss = self.head_ct_loss(x[cut_labels != 0, :], cut_labels[cut_labels != 0])
        if torch.any(torch.isnan(center_update)):
            center_update = 0
        if torch.any(torch.isnan(head_ct_loss)):
            head_ct_loss = 0
        return center_update, head_ct_loss


def fix_eval_modules(eval_modules):
    for module in eval_modules:
        for _, param in module.named_parameters():
            param.requires_grad = False
        # DO NOT use module.eval(), otherwise the module will be in the test mode,
        # i.e., all self.training condition is set to False


def def_retrain_modules(train_modules):
    for module in train_modules:
        for _, param in module.named_parameters():
            param.requires_grad = True
        # DO NOT use module.eval(), otherwise the module will be in the test mode,
        # i.e., all self.training condition is set to False


def run_val(cfg, model, val_data_loaders, distributed, logger):
    if distributed:
        model = model.module
    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    if cfg.MODEL.RELATION_ON:
        iou_types = iou_types + ("relations",)
    if cfg.MODEL.ATTRIBUTE_ON:
        iou_types = iou_types + ("attributes",)


    dataset_names = cfg.DATASETS.VAL

    val_result = []
    for dataset_name, val_data_loader in zip(dataset_names, val_data_loaders):
        dataset_result = inference(
            cfg,
            model,
            val_data_loader,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=None,
            logger=logger,
        )

        synchronize()
        val_result.append(dataset_result)
    # support for multi gpu distributed testing
    gathered_result = all_gather(torch.tensor(dataset_result).cpu())
    gathered_result = [t.view(-1) for t in gathered_result]
    gathered_result = torch.cat(gathered_result, dim=-1).view(-1)
    valid_result = gathered_result[gathered_result >= 0]
    val_result = float(valid_result.mean())
    del gathered_result, valid_result

    return val_result


def run_test(cfg, model, distributed, logger):
    if distributed:
        model = model.module
    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    if cfg.MODEL.RELATION_ON:
        iou_types = iou_types + ("relations",)
    if cfg.MODEL.ATTRIBUTE_ON:
        iou_types = iou_types + ("attributes",)
    output_folders = [None] * len(cfg.DATASETS.TEST)

    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    data_loaders_val = make_data_loader(cfg, mode='test', is_distributed=distributed)
    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        inference(
            cfg,
            model,
            data_loader_val,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
            logger=logger,
        )
        synchronize()

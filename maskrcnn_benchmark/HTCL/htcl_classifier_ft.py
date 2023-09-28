import torch
from maskrcnn_benchmark.modeling.utils import cat
import numpy as np
from torch import nn

import argparse
import os
import time
import datetime
import pickle

from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.data.build import make_data_sampler, make_batch_data_sampler
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.utils.miscellaneous import mkdir, save_config
from maskrcnn_benchmark.utils.logger import setup_logger, debug_print
from maskrcnn_benchmark.utils.comm import synchronize, get_rank, all_gather, is_main_process
from maskrcnn_benchmark.solver import make_optimizer
from maskrcnn_benchmark.solver import make_lr_scheduler
from maskrcnn_benchmark.utils.metric_logger import MetricLogger
from maskrcnn_benchmark.utils.checkpoint import clip_grad_norm
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.modeling.roi_heads.relation_head.utils_relation import layer_init
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.HTCL.htcl_feats_dataset import HTCL_Feats_Dataset
from maskrcnn_benchmark.engine.trainer import reduce_loss_dict
from maskrcnn_benchmark.solver.lr_scheduler import WarmupMultiStepLR, WarmupReduceLROnPlateau
# See if we can use apex.DistributedDataParallel instead of the torch default,
# and enable mixed-precision via apex.amp
# try:
#     from apex import amp
# except ImportError:
#     raise ImportError('Use APEX for multi-precision via apex.amp')


class Classifier(nn.Module):
    def __init__(self, config):
        super(Classifier, self).__init__()
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        self.LAST_FEATS_DIM = config.MODEL.ROI_RELATION_HEAD.LAST_FEATS_DIM
        self.cfg = config
        self.sigmoid = nn.Sigmoid()
        self.fine_cls = nn.Linear(self.LAST_FEATS_DIM, self.num_rel_cls)
        layer_init(self.fine_cls, xavier=True)

    def forward(self, feats, rel_dist0=None, logit_wt=None):
        if self.cfg.MODEL.ft_with_dist0:
            refine_dist = self.fine_cls(feats)
            refine_dist = (refine_dist - refine_dist.mean(dim=1).reshape(-1, 1)) / refine_dist.std(dim=1).reshape(-1, 1)
            rel_dist = rel_dist0 * self.sigmoid(logit_wt) + refine_dist * (1 - self.sigmoid(logit_wt))
            return rel_dist
        else:
            refine_dist = self.fine_cls(feats)
            refine_dist = (refine_dist - refine_dist.mean(dim=1).reshape(-1, 1)) / refine_dist.std(dim=1).reshape(-1, 1)
            return refine_dist


def make_classifier_data_loader(cfg, dataset=None, batch_size=None, shuffle=True, start_iter=0, is_distributed=False):
    images_per_gpu = batch_size
    num_iters = cfg.SOLVER.Classifier_max_iter
    sampler = make_data_sampler(dataset, shuffle, is_distributed)
    batch_sampler = make_batch_data_sampler(
        dataset, sampler, False, images_per_gpu, num_iters, start_iter
    )

    num_workers = cfg.DATALOADER.NUM_WORKERS
    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
    )
    return data_loader


def sgg_model_load_parameterss(model, classifier, distributed):
    if distributed:
        classifier = classifier.module
    with torch.no_grad():
        model.roi_heads.relation.predictor.fine_cls.weight.copy_(classifier.fine_cls.weight, non_blocking=True)
        model.roi_heads.relation.predictor.fine_cls.bias.copy_(classifier.fine_cls.bias, non_blocking=True)
    return model


def fix_eval_modules(eval_modules):
    for module in eval_modules:
        for _, param in module.named_parameters():
            param.requires_grad = False


def run_val(cfg, model, val_data_loaders, distributed, logger):
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
    # torch.cuda.empty_cache()
    return val_result


def run_test(cfg, model, distributed, logger):
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
    if cfg.CLASSIFIER_OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.CLASSIFIER_OUTPUT_DIR, "inference", dataset_name)
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



def htcl_classifier_ft(cfg, local_rank, distributed, logger):
    debug_print(logger, 'prepare training')
    classifier = Classifier(cfg)
    debug_print(logger, 'end model construction')
    device = torch.device(cfg.MODEL.DEVICE)
    classifier.to(device)

    num_batch = cfg.SOLVER.Classifier_batch
    cls_optimizer = make_optimizer(cfg, classifier, logger, slow_heads=[], slow_ratio=10.0,
                                   rl_factor=float(num_batch))
    scheduler = WarmupMultiStepLR(cls_optimizer, cfg.SOLVER.cls_ft_STEPS, 0.1, warmup_factor=0.01, warmup_iters=2000)
    debug_print(logger, 'end optimizer and shcedule')

    # Initialize mixed-precision training
    # use_mixed_precision = cfg.DTYPE == "float16"
    # amp_opt_level = 'O1' if use_mixed_precision else 'O0'
    # classifier, cls_optimizer = amp.initialize(classifier, cls_optimizer, opt_level= amp_opt_level,verbosity=0)

    if distributed:
        classifier = torch.nn.parallel.DistributedDataParallel(
            classifier, device_ids=[local_rank], output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
            find_unused_parameters=True,
        )
    debug_print(logger, 'end distributed')

    arguments = {}
    arguments["iteration"] = 0
    arguments["last_val_results"] = 0
    arguments["best_val_results"] = 0

    output_dir = cfg.CLASSIFIER_OUTPUT_DIR
    save_to_disk = get_rank() == 0

    checkpointer = DetectronCheckpointer(cfg, classifier, cls_optimizer,scheduler, save_dir= output_dir, save_to_disk= save_to_disk,logger=logger, custom_scheduler=True)

    if checkpointer.has_checkpoint():
        extra_checkpoint_data = checkpointer.load()
        arguments.update(extra_checkpoint_data)
    else:
        load_mapping = {"fine_cls": "roi_heads.relation.predictor.fine_cls"}
        checkpointer.load(cfg.MODEL.Feature_Generation_MODEL, with_optim=False, load_mapping=load_mapping)

    sgg_model = build_detection_model(cfg)
    eval_modules = (sgg_model.rpn, sgg_model.backbone, sgg_model.roi_heads,)
    fix_eval_modules(eval_modules)
    sgg_model.to(device)

    # use_mixed_precision = cfg.DTYPE == "float16"
    # amp_opt_level = 'O1' if use_mixed_precision else 'O0'
    # sgg_model = amp.initialize(sgg_model, opt_level=amp_opt_level)

    checkpointer_sgg = DetectronCheckpointer(cfg, sgg_model, logger=logger)
    checkpointer_sgg.load(cfg.MODEL.Feature_Generation_MODEL, with_optim=False)

    sgg_model = sgg_model_load_parameterss(sgg_model, classifier, distributed)

    # # # # # # # # # # # # # # #
    if cfg.MODEL.ft_with_dist0:
        logit_wt = sgg_model.roi_heads.relation.predictor.logit_wt

    data_set = HTCL_Feats_Dataset(cfg, logger)
    train_data_loader = make_classifier_data_loader(cfg, dataset=data_set, batch_size=num_batch,
                                                    shuffle=True, start_iter=arguments["iteration"], is_distributed=distributed)
    val_data_loaders = make_data_loader(cfg, mode='val', is_distributed=distributed,)
    debug_print(logger, 'end dataloader')

    if cfg.SOLVER.PRE_VAL:
        logger.info("Validate before training")
        run_val(cfg, sgg_model, val_data_loaders, distributed, logger)

    logger.info("Start training")
    ce_loss = nn.CrossEntropyLoss()
    meters = MetricLogger(delimiter="  ")
    max_iter = len(train_data_loader)
    start_iter = arguments["iteration"]

    start_training_time = time.time()
    end = time.time()
    print_first_grad = True

    for iteration, batch in enumerate(train_data_loader, start_iter):
        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration
        classifier.train()

        if cfg.MODEL.ft_with_dist0:
            feats, rels, rel_dist0 = batch
            rel_dist0 = rel_dist0.to(device)
            feats = feats.to(device)
            rels = rels.to(device)
            rel_dist = classifier(feats, rel_dist0, logit_wt)
        else:
            feats, rels = batch
            feats = feats.to(device)
            rels = rels.to(device)
            rel_dist = classifier(feats)
        loss_ce = ce_loss(rel_dist.float(), rels.long())
        loss_relation = dict(loss_ce=loss_ce)

        losses = sum(loss for loss in loss_relation.values())
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_relation)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        meters.update(loss=losses_reduced, **loss_dict_reduced)
        cls_optimizer.zero_grad()

        # Note: If mixed precision is not used, this ends up doing nothing
        # Otherwise apply loss scaling for mixed-precision recipe
        # with amp.scale_loss(losses, cls_optimizer) as scaled_losses:
        #     scaled_losses.backward()

        losses.backward()
        # add clip_grad_norm from MOTIFS, tracking gradient, used for debug
        verbose = (iteration % cfg.SOLVER.Classifier_checkpoint_period) == 0 \
                  or print_first_grad  # print grad or not
        print_first_grad = False
        clip_grad_norm([(n, p) for n, p in classifier.named_parameters() if p.requires_grad],
                       max_norm=cfg.SOLVER.GRAD_NORM_CLIP, logger=logger, verbose=verbose, clip=True)
        cls_optimizer.step()
        scheduler.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % (cfg.SOLVER.Classifier_checkpoint_period/10) == 0:
            logger.info(meters.delimiter.join(
                [
                    "eta: {eta}",
                    "iter: {iter}",
                    "{meters}",
                    "lr: {lr:.6f}",
                    "max_iter: {max_iter}",
                ]
            ).format(
                eta=eta_string,
                iter=iteration,
                meters=str(meters),
                lr=cls_optimizer.param_groups[-1]["lr"],
                max_iter=max_iter,
            ))

        if iteration % cfg.SOLVER.Classifier_checkpoint_period == 0:
            logger.info("Start validating")
            sgg_model = sgg_model_load_parameterss(sgg_model, classifier, distributed)
            val_result = run_val(cfg, sgg_model, val_data_loaders, distributed, logger)
            arguments["last_val_results"] = val_result
            if arguments["last_val_results"] >= arguments["best_val_results"]:
                arguments["best_val_results"] = arguments["last_val_results"]

            logger.info("Validation Result: %.4f" % val_result)
            checkpointer.save("model_{:07d}".format(iteration), **arguments)

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )

    with open(output_dir + "/End_traning", 'w') as f:
        f.write('End classifier training')

    if not cfg.TEST.Skip_test:
        checkpointer.load()
        sgg_model = sgg_model_load_parameterss(sgg_model, classifier, distributed)
        run_test(cfg, sgg_model, distributed, logger)

    return classifier, sgg_model
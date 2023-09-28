# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Basic training script for PyTorch
"""
# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip
import argparse
import os
import torch
import logging

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir, save_config
from maskrcnn_benchmark.HTCL.htcl_train_sgg_net import htcl_train_sgg_net
from maskrcnn_benchmark.HTCL.htcl_classifier_ft import htcl_classifier_ft
from maskrcnn_benchmark.HTCL.htcl_feats_generation import htcl_feats_generation
import numpy as np
import random

def main():
    parser = argparse.ArgumentParser(description="PyTorch Relation Detection Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--my_opts",
        default="",
        metavar="FILE",
        help="path to my config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)

    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_file(args.my_opts)
    cfg.merge_from_list(args.opts)
    if cfg.OUTPUT_DIR:
        mkdir(cfg.OUTPUT_DIR)

    if not os.path.exists(cfg.OUTPUT_DIR + '/End_traning'):
        with open(args.my_opts, "r") as f:
            f_opts = f.read()
        with open(cfg.OUTPUT_DIR+ "/my_opts.yaml", 'w') as f:
            f.write(f_opts)
        logger = setup_logger("maskrcnn_benchmark", cfg.OUTPUT_DIR, get_rank())
        logger.info("Using {} GPUs".format(num_gpus))
        logger.info(args)

        logger.info("Collecting env info (might take some time)")
        logger.info("\n" + collect_env_info())

        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, "r") as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
        logger.info("Running with config:\n{}".format(cfg))

        output_config_path = os.path.join(cfg.OUTPUT_DIR, 'config.yml')
        logger.info("Saving config into: {}".format(output_config_path))
        save_config(cfg, output_config_path)

        # train sgg model
        htcl_train_sgg_net(cfg, args.local_rank, args.distributed, logger)

    best_model_path = os.path.join(cfg.OUTPUT_DIR, "last_checkpoint")
    with open(best_model_path, "r") as f_saved:
        cfg.MODEL.Feature_Generation_MODEL = f_saved.read().strip()

    if Feature_Generation_conditon(cfg):
        if not 'logger' in locals().keys():
            logger = setup_logger("maskrcnn_benchmark", cfg.OUTPUT_DIR, get_rank())
        cfg.MODEL.Feature_Generation_Mode = True
        htcl_feats_generation(cfg, args.local_rank, args.distributed, logger)
        cfg.MODEL.Feature_Generation_Mode = False

    if cfg.CLASSIFIER_OUTPUT_DIR:
        mkdir(cfg.CLASSIFIER_OUTPUT_DIR)
    if not os.path.exists(cfg.CLASSIFIER_OUTPUT_DIR + '/End_traning'):
        with open(args.my_opts, "r") as f:
            f_opts = f.read()
        with open(cfg.CLASSIFIER_OUTPUT_DIR + "/my_opts.yaml", 'w') as f:
            f.write(f_opts)

        cfg.TEST.Val_mode = 'mRecall'
        logger_cls = setup_logger("classifier", cfg.CLASSIFIER_OUTPUT_DIR, get_rank())
        logger_cls.info("Using {} GPUs".format(num_gpus))
        logger_cls.info(args)

        logger_cls.info("Collecting env info (might take some time)")
        logger_cls.info("\n" + collect_env_info())

        logger_cls.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, "r") as cf:
            config_str = "\n" + cf.read()
            logger_cls.info(config_str)
        logger_cls.info("Running with config:\n{}".format(cfg))

        output_config_path = os.path.join(cfg.CLASSIFIER_OUTPUT_DIR, 'config.yml')
        logger_cls.info("Saving config into: {}".format(output_config_path))
        save_config(cfg, output_config_path)

        cfg.SOLVER.BASE_LR = cfg.SOLVER.BASE_LR_ft
        cfg.SOLVER.MAX_ITER = cfg.SOLVER.Classifier_max_iter
        cfg.SOLVER.IMS_PER_BATCH = cfg.SOLVER.Classifier_batch
        htcl_classifier_ft(cfg, args.local_rank, args.distributed, logger_cls)


def Feature_Generation_conditon(cfg):
    conditon = False
    if not os.path.exists(cfg.OUTPUT_DIR_feats + '/bg_feats/') \
            or not os.path.exists(cfg.OUTPUT_DIR_feats + '/fg_feats.pth') \
            or not os.path.exists(cfg.OUTPUT_DIR_feats + '/feats_info.pth'):
        conditon = True
    return conditon


if __name__ == "__main__":

    main()

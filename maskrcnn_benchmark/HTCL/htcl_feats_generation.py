import torch
from maskrcnn_benchmark.modeling.utils import cat
import time
import datetime
import numpy as np
from tqdm import tqdm

from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.utils.miscellaneous import mkdir
from maskrcnn_benchmark.utils.logger import debug_print
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.comm import synchronize,all_gather, is_main_process

# try:
#     from apex import amp
# except ImportError:
#     raise ImportError('Use APEX for multi-precision via apex.amp')


def fix_eval_modules(eval_modules):
    for module in eval_modules:
        for _, param in module.named_parameters():
            param.requires_grad = False

def htcl_feats_generation(cfg, local_rank, distributed, logger):
    debug_print(logger, 'prepare features_generation')

    model = build_detection_model(cfg)
    debug_print(logger, 'end model construction')
    eval_modules = (model.rpn, model.backbone, model.roi_heads,)
    fix_eval_modules(eval_modules)

    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    debug_print(logger, 'end optimizer and shcedule')
    debug_print(logger, 'end distributed')
    arguments = {}
    arguments["iteration"] = 0

    checkpointer = DetectronCheckpointer(cfg, model)
    checkpointer.load(cfg.MODEL.Feature_Generation_MODEL, with_optim=False)
    debug_print(logger, 'end load checkpointer')

    train_data_loader = make_data_loader(
        cfg,
        mode='train',
        is_distributed=distributed,
        start_iter=arguments["iteration"],
    )
    debug_print(logger, 'end dataloader')

    logger.info("Start Features Generation")
    model.eval()

    rel_labels = []
    obj_pair_preds = []
    rel_dists_0 = []
    fg_feats_runing = []
    bg_feats_runing = None
    save_index = 0
    feats_dir = cfg.OUTPUT_DIR_feats + '/bg_feats/'
    mkdir(feats_dir)

    max_bg_feats = cfg.MODEL.max_bg_feats

    for _, batch in enumerate(tqdm(train_data_loader)):
        with torch.no_grad():
            images, targets, images_idx_grp = batch
            targets = [target.to(device) for target in targets]
            results = model(images.to(device), targets)
            last_feats_grp, rel_labels_grp, obj_pair_preds_grp, rel_dists_0_grp = results

            # from ft16 to ft32 in .cpu()
            last_feats_grp = [o.cpu() for o in last_feats_grp]
            rel_labels_grp = [o.cpu()  for o in rel_labels_grp]
            obj_pair_preds_grp = [o.cpu()  for o in obj_pair_preds_grp]
            rel_dists_0_grp = [o.cpu() for o in rel_dists_0_grp]
            synchronize()

            last_feats_grp = all_gather(last_feats_grp)
            rel_labels_grp = all_gather(rel_labels_grp)
            obj_pair_preds_grp = all_gather(obj_pair_preds_grp)
            rel_dists_0_grp = all_gather(rel_dists_0_grp)
            if is_main_process():
                for p1,p2,p3,p4 in zip(last_feats_grp,rel_labels_grp,obj_pair_preds_grp,rel_dists_0_grp):
                    for i in range(len(p1)):
                        rel_labels.append(p2[i].clone().numpy())
                        obj_pair_preds.append(p3[i].clone().numpy())
                        rel_dists_0.append(p4[i].clone().numpy())
                        feats_runing = torch.from_numpy(p1[i].clone().numpy())
                        label_runing = torch.from_numpy(p2[i].clone().numpy())
                        fg_feats_runing.append(feats_runing[label_runing > 0, :])

                        if save_index < max_bg_feats or max_bg_feats == 0:
                            if bg_feats_runing is None:
                                bg_feats_runing = feats_runing[label_runing == 0, :]
                            else:
                                bg_feats_runing = cat([bg_feats_runing, feats_runing[label_runing == 0, :]])
                if save_index < max_bg_feats or max_bg_feats == 0:
                    while bg_feats_runing.shape[0] >= 100:
                        torch.save(bg_feats_runing[0:100, :].clone(), feats_dir + str(save_index) + '_' + str(save_index + 99))
                        bg_feats_runing = bg_feats_runing[100:, :]
                        save_index = save_index + 100

    del last_feats_grp, rel_labels_grp, obj_pair_preds_grp

    if is_main_process():
        # assert bg_feats_runing.shape[0] < 100
        if max_bg_feats == 0:
            torch.save(bg_feats_runing.clone(), feats_dir + str(save_index) + '_' + str(save_index +bg_feats_runing.shape[0]-1))

        rels_labels_all = cat([torch.from_numpy(rel_labels[i]) for i in range(len(rel_labels))])
        obj_pair_all_preds = cat([torch.from_numpy(obj_pair_preds[i]) for i in range(len(obj_pair_preds))], dim=0)
        rel_dists_0_all =  cat([torch.from_numpy(rel_dists_0[i]) for i in range(len(rel_dists_0))], dim=0)
        del rel_labels, obj_pair_preds, rel_dists_0

        fg_index = np.where(rels_labels_all != 0)[0]
        if max_bg_feats == 0:
            bg_index = np.where(rels_labels_all == 0)[0]
        else:
            bg_index = np.where(rels_labels_all == 0)[0][:max_bg_feats]

        fg_feats = cat([fg_feats_runing[i] for i in range(len(fg_feats_runing))])
        fg_rels = rels_labels_all[fg_index]
        fg_obj_pair_preds = obj_pair_all_preds[fg_index, :]
        fg_rel_dists_0 = rel_dists_0_all[fg_index, :]

        bg_rels = rels_labels_all[bg_index]
        bg_obj_pair_preds = obj_pair_all_preds[bg_index,:]
        bg_rel_dists_0 = rel_dists_0_all[bg_index, :]

        rels_labels_fg_bg = cat([fg_rels, bg_rels])
        obj_pair_preds_fg_bg = cat([fg_obj_pair_preds, bg_obj_pair_preds], dim=0)
        rel_dists_0_fg_bg = cat([fg_rel_dists_0, bg_rel_dists_0], dim=0)

        torch.save(fg_feats, cfg.OUTPUT_DIR_feats + '/fg_feats.pth')
        del fg_feats

        feats_info = {}
        feats_info['rel_labels'] = rels_labels_fg_bg
        feats_info['obj_pair_preds'] = obj_pair_preds_fg_bg
        feats_info['rel_dists_0'] = rel_dists_0_fg_bg
        torch.save(feats_info, cfg.OUTPUT_DIR_feats + '/feats_info.pth')

    print('End Feature_Generation_Mode')
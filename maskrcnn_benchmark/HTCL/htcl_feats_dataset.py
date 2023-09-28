import torch
from maskrcnn_benchmark.modeling.utils import cat
import numpy as np
import os
import pickle
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.utils.comm import get_rank, synchronize


class HTCL_Feats_Dataset(torch.utils.data.Dataset):
    def __init__(self, cfg, logger=None):
        feats_info_path = cfg.OUTPUT_DIR_feats + '/feats_info.pth'
        self.bg_feats_path = cfg.OUTPUT_DIR_feats + '/bg_feats/'
        feats_info = torch.load(feats_info_path)
        self.fg_feats = torch.load(cfg.OUTPUT_DIR_feats + '/fg_feats.pth')
        self.cfg = cfg
        self.Feats_resampling = cfg.MODEL.Feats_resampling

        self.rels_labels = feats_info['rel_labels']
        self.obj_pair = feats_info['obj_pair_preds'].long()
        if 'rel_dists_0' in feats_info:
            self.rel_dists_0 = feats_info['rel_dists_0']

        self.idx_list = list(range(len(self.rels_labels)))
        self.max_idx_list = max(self.idx_list)
        self.repeat_dict = None

        if self.Feats_resampling:
            repeat_dict_dir = os.path.join(cfg.CLASSIFIER_OUTPUT_DIR, "repeat_dict.pkl")
            if os.path.exists(repeat_dict_dir):
                print("load repeat_dict from " + repeat_dict_dir)
                with open(repeat_dict_dir, 'rb') as f:
                    repeat_dict = pickle.load(f)
            else:
                if get_rank() == 0:
                    print("generate repeat_dict to " + repeat_dict_dir)
                    repeat_dict = feats_repeat_dict_generation(self, logger)
                    with open(repeat_dict_dir, "wb") as f:
                        pickle.dump(repeat_dict, f)
                synchronize()
                with open(repeat_dict_dir, 'rb') as f:
                    repeat_dict = pickle.load(f)
            self.repeat_dict = repeat_dict

            self.idx_list = self.repeat_dict[:, -1].tolist()
            self.max_idx_list = max(self.idx_list)

        self.fg_count = len(np.where(self.rels_labels != 0)[0])
        self.bg_count = len(np.where(self.rels_labels == 0)[0]) - 1

    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, index):
        if self.repeat_dict is not None:
            index = self.idx_list[index]
        rels = self.rels_labels[index]
        if rels != 0:
            feats = self.fg_feats[index, :]
        else:
            bg_id = index - self.fg_count
            if 100 * int(self.bg_count/100) <= bg_id:
                feats_grp = torch.load(self.bg_feats_path + str(100 * int(bg_id / 100)) + '_' + str(self.bg_count))
            else:
                feats_grp = torch.load(self.bg_feats_path + str(100*int(bg_id/100)) + '_' + str(100*(int(bg_id/100)+1)-1))
            feats = feats_grp[bg_id % 100, :]
        if self.cfg.MODEL.ft_with_dist0:
            rel_dists_0 = self.rel_dists_0[index, :]
            return feats, rels, rel_dists_0
        return feats, rels


def feats_repeat_dict_generation(dataset, logger):
    bg = dataset.rels_labels.numpy() == 0
    fg = dataset.rels_labels.numpy() != 0

    num_class = cfg.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
    rels_fg = dataset.rels_labels[fg].numpy()
    obj_pair_fg = dataset.obj_pair[fg, :].numpy()
    # [rel,o1,o2,feat_i]
    rel_list = [np.array([[0, 0, 0, 0]], dtype=int) for x in range(num_class)]
    fg_index = np.where(dataset.rels_labels.numpy() != 0)[0]

    F_c = np.zeros(num_class, dtype=int)

    bg_dict = np.zeros([bg.sum(), 4], dtype=int)
    bg_dict[:, 3] = np.array(np.where(dataset.rels_labels.numpy() == 0))

    rel_list_info_dir = os.path.join(cfg.CLASSIFIER_OUTPUT_DIR, "rel_list_info.pkl")
    if os.path.exists(rel_list_info_dir):
        print("load rel_list_info from " + rel_list_info_dir)
        with open(rel_list_info_dir, 'rb') as f:
            rel_list_info = pickle.load(f)
        rel_list = rel_list_info['rel_list']
        F_c = rel_list_info['F_c']
    else:
        print("generate rel_list_info to " + rel_list_info_dir)
        for i in range(len(rels_fg)):
            rel_list[rels_fg[i]] = np.concatenate((rel_list[rels_fg[i]], np.array(
                [[rels_fg[i], obj_pair_fg[i, 0], obj_pair_fg[i, 1], fg_index[i]]])), axis=0)
            F_c[rels_fg[i]] += 1
        rel_list_info = {}
        rel_list_info['rel_list'] = rel_list
        rel_list_info['F_c'] = F_c
        with open(rel_list_info_dir, "wb") as f:
            pickle.dump(rel_list_info, f)
        print("end generate rel_list_info to " + rel_list_info_dir)

    F_c[0] = bg.sum()
    if dataset.cfg.MODEL.Rels_each_class == 0:
        num_per_cls = max(F_c[1:])
    else:
        num_per_cls = dataset.cfg.MODEL.Rels_each_class

    fg_repeat_dict = {}
    for i_rel in range(1, num_class):
        each_relation_dict = rel_list[i_rel][1:, :]
        if each_relation_dict.shape[0] > 0:
            fg_repeat_dict[i_rel] = each_relation_dict[np.random.randint(0, each_relation_dict.shape[0], num_per_cls)]

    fg_repeat_dict = np.vstack([fg_repeat_dict[k] for k in fg_repeat_dict])
    repeat_dict = np.vstack([fg_repeat_dict, bg_dict[:cfg.MODEL.max_bg_feats,:]])
    return repeat_dict



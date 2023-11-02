# HTCL

The repository is the implementation of paper "Head-Tail Cooperative Learning Network for Unbiased Scene Graph Generation".

## Installation

See [INSTALL.md](./INSTALL.md) for installation instructions.

## Dataset

See [DATASET.md](./DATASET.md) for instructions of dataset preprocessing.

## Training

Our experiments are conducted on one NVIDIA GeForce RTX 3090, If you want to run it on your own device, please refer to [Scene-Graph-Benchmark.pytorch](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch).

We provide [scripts](./scripts/PENET_HTCL_preds.sh) for training the models:
```
export CUDA_VISIBLE_DEVICES=0  
export NUM_GPU=1  
  
SCRIPTNAME=$(basename "$0")  
MODEL_NAME='PENET_HTCL_preds'  
mkdir /opt/data/private/checkpoints/HTCL/${MODEL_NAME}/  
cp ./${SCRIPTNAME} /opt/data/private/checkpoints/HTCL/${MODEL_NAME}/  
  
python3 tools/HTCL_main.py \  
  --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \  
  --my_opts "./maskrcnn_benchmark/HTCL/HTCL_base_opts.yaml" \  
  MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \  
  MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \  
  MODEL.ROI_RELATION_HEAD.PREDICTOR PENET_HTCL \  
  OUTPUT_DIR /opt/data/private/checkpoints/HTCL/${MODEL_NAME} \  
  OUTPUT_DIR_feats /opt/data/private/feats/HTCL/${MODEL_NAME} \  
  CLASSIFIER_OUTPUT_DIR /opt/data/private/checkpoints/HTCL/${MODEL_NAME}/cls_ft
```

Please modify the path and parameters in `./maskrcnn_benchmark/HTCL/HTCL_base_opts.yaml` to fit your own device and task.

If you have any questions, please contact me (`wlei0618@foxmail.com`).

## Acknowledgement

The code is implemented based on [Scene-Graph-Benchmark.pytorch](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch) and [PENET](https://github.com/VL-Group/PENET).

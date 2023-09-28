source /opt/data/private/envs/conda/bin/activate HTCL
cd /opt/data/private/projects/HTCL

export CUDA_VISIBLE_DEVICES=6
export NUM_GPU=1
echo "TRAINING"

SCRIPTNAME=$(basename "$0")
MODEL_NAME='Motifs_HTCL_sggen'
mkdir /opt/data/private/checkpoints/HTCL/${MODEL_NAME}/
cp ./${SCRIPTNAME} /opt/data/private/checkpoints/HTCL/${MODEL_NAME}/

python3 tools/HTCL_main.py \
  --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
  --my_opts "./maskrcnn_benchmark/HTCL/HTCL_base_opts.yaml" \
  MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \
  MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
  MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor_HTCL \
  OUTPUT_DIR /opt/data/private/checkpoints/HTCL/${MODEL_NAME} \
  OUTPUT_DIR_feats /opt/data/private/feats/HTCL/${MODEL_NAME} \
  CLASSIFIER_OUTPUT_DIR /opt/data/private/checkpoints/HTCL/${MODEL_NAME}/cls_ft \
  SOLVER.PRE_VAL True

#rm -r OUTPUT_DIR_feats /opt/data/private/feats/HTCL/${MODEL_NAME}

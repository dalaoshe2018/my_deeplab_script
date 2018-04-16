# Set up the working directories.
LIP_FOLDER="/home/liyongbin/LIP"
WORK_DIR="/home/liyongbin/models/research/deeplab"
DATASET="LIP"

set -e
cd "${WORK_DIR}"
cd ..
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

CURRENT_DIR=$(pwd)
echo ${CURRENT_DIR}

EXP_FOLDER="exp/train_on_trainval_set"

INIT_FOLDER="${WORK_DIR}/init_models"
TRAIN_LOGDIR="${WORK_DIR}/${DATASET}/train"
EVAL_LOGDIR="${WORK_DIR}/${DATASET}/eval"
VIS_LOGDIR="${WORK_DIR}/${DATASET}/vis"
EXPORT_DIR="${WORK_DIR}/${DATASET}/export"

mkdir -p "${INIT_FOLDER}"
mkdir -p "${TRAIN_LOGDIR}"
mkdir -p "${EVAL_LOGDIR}"
mkdir -p "${VIS_LOGDIR}"
mkdir -p "${EXPORT_DIR}"

cd "${CURRENT_DIR}"

LIP_DATASET="${LIP_FOLDER}/train_img/tfcord"
# Train 10 iterations.
let NUM_ITERATIONS=7500*100
python3.5 "${WORK_DIR}"/train.py \
  --logtostderr \
  --train_split="train_id" \
  --dataset="LIP" \
  --log_steps=1000 \
  --model_variant="xception_65" \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --initialize_last_layer=true\
  --save_interval_secs=3750 \
  --decoder_output_stride=4 \
  --train_crop_size=321 \
  --train_crop_size=321 \
  --train_batch_size=4 \
  --training_number_of_steps="${NUM_ITERATIONS}" \
  --fine_tune_batch_norm=true \
  --tf_initial_checkpoint="${INIT_FOLDER}/model.ckpt" \
  --train_logdir="${TRAIN_LOGDIR}" \
  --dataset_dir="${LIP_DATASET}"


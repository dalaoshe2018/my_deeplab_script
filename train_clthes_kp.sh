
# Set up the working directories.
HOME_DIR="/home/liyongbin"
WORK_DIR="${HOME_DIR}/models/research/deeplab"
DATASET="CLOTHES"

set -e
cd "${WORK_DIR}"
cd ..
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

CURRENT_DIR=$(pwd)
echo ${CURRENT_DIR}


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

CLOTHES_DATASET="${HOME_DIR}/train_clothes_img/tfcord"
# Train 10 iterations.
let NUM_ITERATIONS=7500*100
python3.5 "${WORK_DIR}"/train.py \
  --logtostderr \
  --train_split="train" \
  --dataset=${DATASET} \
  --log_steps=100 \
  --model_variant="xception_65" \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --initialize_last_layer=true\
  --save_interval_secs=300 \
  --decoder_output_stride=4 \
  --train_crop_size=520 \
  --train_crop_size=520 \
  --train_batch_size=4 \
  --training_number_of_steps="${NUM_ITERATIONS}" \
  --fine_tune_batch_norm=true \
  --tf_initial_checkpoint="${INIT_FOLDER}/model.ckpt" \
  --train_logdir="${TRAIN_LOGDIR}" \
  --dataset_dir="${CLOTHES_DATASET}"


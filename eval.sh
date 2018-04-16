# Set up the working directories.
HOME_DIR="/home/liyongbin/"
LIP_FOLDER="${HOME_DIR}/LIP"
WORK_DIR="${HOME_DIR}/models/research/deeplab"
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

# Copy locally the trained checkpoint as the initial checkpoint.
#TF_INIT_ROOT="http://download.tensorflow.org/models"
#TF_INIT_CKPT="deeplabv3_pascal_train_aug_2018_01_04.tar.gz"
#cd "${INIT_FOLDER}"
#wget -nd -c "${TF_INIT_ROOT}/${TF_INIT_CKPT}"
#tar -xf "${TF_INIT_CKPT}"
cd "${CURRENT_DIR}"

LIP_DATASET="${LIP_FOLDER}/train_img/tfcord"
# Train 10 iterations.
let NUM_ITERATIONS=7500*100

## will take a while.
## Using the provided checkpoint, one should expect mIOU=82.20%.
python3.5 "${WORK_DIR}"/eval.py \
  --logtostderr \
  --eval_split="val_id" \
  --dataset="LIP" \
  --model_variant="xception_65" \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --decoder_output_stride=4 \
  --eval_crop_size=800 \
  --eval_crop_size=800 \
  --checkpoint_dir="${TRAIN_LOGDIR}" \
  --eval_logdir="${EVAL_LOGDIR}" \
  --dataset_dir="${LIP_DATASET}" \
  --max_number_of_evaluations=1

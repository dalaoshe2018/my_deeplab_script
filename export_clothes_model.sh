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

# Copy locally the trained checkpoint as the initial checkpoint.
#TF_INIT_ROOT="http://download.tensorflow.org/models"
#TF_INIT_CKPT="deeplabv3_pascal_train_aug_2018_01_04.tar.gz"
#cd "${INIT_FOLDER}"
#wget -nd -c "${TF_INIT_ROOT}/${TF_INIT_CKPT}"
#tar -xf "${TF_INIT_CKPT}"
cd "${CURRENT_DIR}"

CLOTHES_DATASET="${HOME_DIR}/train_clothes_img/tfcord"
# Train 10 iterations.
let NUM_ITERATIONS=356

## will take a while.
## Using the provided checkpoint, one should expect mIOU=82.20%.


CKPT_PATH="${TRAIN_LOGDIR}/model.ckpt-${NUM_ITERATIONS}"
EXPORT_PATH="${EXPORT_DIR}/frozen_inference_graph.pb"

python3.5 "${WORK_DIR}"/export_model.py \
  --logtostderr \
  --checkpoint_path="${CKPT_PATH}" \
  --export_path="${EXPORT_PATH}" \
  --model_variant="xception_65" \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --decoder_output_stride=4 \
  --num_classes=25 \
  --crop_size=520 \
  --crop_size=520 \
  --inference_scales=1.0

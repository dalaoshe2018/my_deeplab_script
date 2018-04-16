from Deeplab_LIP_Model import *
from PIL import Image


import tensorflow as tf
from matplotlib import gridspec
from matplotlib import pyplot as plt
from dataset import *
#from deeplab import common
#from deeplab import model
#from deeplab.datasets import segmentation_dataset
#from deeplab.utils import input_generator
#from deeplab.utils import save_annotation

slim = tf.contrib.slim

flags = tf.app.flags

FLAGS = flags.FLAGS



# The folder where semantic segmentation predictions are saved.
_SEMANTIC_PREDICTION_SAVE_FOLDER = 'segmentation_results'

# The folder where raw semantic segmentation predictions are saved.
_RAW_SEMANTIC_PREDICTION_SAVE_FOLDER = 'raw_segmentation_results'

# The format to save image.
_IMAGE_FORMAT = '%06d_image'

# The format to save prediction
_PREDICTION_FORMAT = '%06d_prediction'

# To evaluate Cityscapes results on the evaluation server, the labels used
# during training should be mapped to the labels for evaluation.
_CITYSCAPES_TRAIN_ID_TO_EVAL_ID = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22,
                                   23, 24, 25, 26, 27, 28, 31, 32, 33]


_TARBALL_NAME = 'deeplab_model_xcep.tar.gz'
model_dir = os.getcwd()
model_path = os.path.join(model_dir, _TARBALL_NAME)


LABEL_NAMES = np.asarray([
    "0.Background",
    "1.Hat",
    "2.Hair",
    "3.Glove",
    "4.Sunglasses",
   "5.UpperClothes",
   "6.Dress",
   "7.Coat",
   "8.Socks",
   "9.Pants",
   "10.Jumpsuits",
   "11.Scarf",
   "12.Skirt",
   "13.Face",
   "14.Left-arm",
   "15.Right-arm",
   "16.Left-leg",
   "17.Right-leg",
   "18.Left-shoe",
   "19.Right-shoe",
   "20", '21', '22', '23', '24', '25', '26', '27'
])

def create_pascal_label_colormap():
    """Creates a label colormap used in PASCAL VOC segmentation benchmark.

    Returns:
    A Colormap for visualizing segmentation results.
    """
    colormap = np.zeros((256, 3), dtype=int)
    ind = np.arange(256, dtype=int)

    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((ind >> channel) & 1) << shift
        ind >>= 3

    return colormap
def label_to_color_image(label):
    """Adds color defined by the dataset colormap to the label.

    Args:
    label: A 2D array with integer type, storing the segmentation label.

    Returns:
    result: A 2D array with floating type. The element of the array
      is the color indexed by the corresponding element in the input label
      to the PASCAL color map.

    Raises:
    ValueError: If label is not of rank 2 or its value is larger than color
      map maximum entry.
    """
    if label.ndim != 2:
        raise ValueError('Expect 2-D input label')

    colormap = create_pascal_label_colormap()

    if np.max(label) >= len(colormap):
        raise ValueError('label value too large.')

    return colormap[label]

FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)


def vis_segmentation(image, seg_map):
    """Visualizes input image, segmentation map and overlay view."""
    print("Begin vis")
    plt.figure(figsize=(15, 5))
    print("figure over")
    grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])

    plt.subplot(grid_spec[0])
    plt.imshow(image)
    plt.axis('off')
    plt.title('input image')

    plt.subplot(grid_spec[1])
    seg_image = label_to_color_image(seg_map).astype(np.uint8)
    plt.imshow(seg_image)
    plt.axis('off')
    plt.title('segmentation map')

    plt.subplot(grid_spec[2])
    plt.imshow(image)
    plt.imshow(seg_image, alpha=0.7)
    plt.axis('off')
    plt.title('segmentation overlay')

    unique_labels = np.unique(seg_map)
    ax = plt.subplot(grid_spec[3])
    plt.imshow(
      FULL_COLOR_MAP[unique_labels].astype(np.uint8), interpolation='nearest')
    ax.yaxis.tick_right()
    plt.yticks(range(len(unique_labels)), LABEL_NAMES[unique_labels])
    plt.xticks([], [])
    ax.tick_params(width=0.0)
    plt.grid('off')
    plt.show()


def main(unused_argv):
    MODEL_DIR = "/home/liyongbin/models/research/deeplab/LIP/train"
    model = DeeplabLIPModel(
            mode="training", 
            config="",
            model_dir=MODEL_DIR,
            model_label=37500)
    #model.load_model()
    #model.load_from_meta()
    model.load_from_pb(model_path)

    dataset_file = "/home/liyongbin/LIP/train_img/train_id.txt"
    img_folder = "/home/liyongbin/LIP/train_img/train_images"
    seg_folder = "/home/liyongbin/LIP/train_img/train_parse_annotation/train_segmentations"
    dataset_train = DataSet(
            dataset_file,
            image_folder=img_folder,
            gt_seg_folder=seg_folder)
    
    #model.train_model(dataset=dataset_train)

    img, seg = dataset_train.get_next_batch_datas()
    #vis_segmentation(img[0], seg[0])
    img,seg = model.detect(img)
    vis_segmentation(img[0], seg)

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
    tf.app.run()

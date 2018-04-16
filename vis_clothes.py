from Deeplab_LIP_Model import *
from PIL import Image


import tensorflow as tf
from matplotlib import gridspec
from matplotlib import pyplot as plt
from dataset import *


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


def save_res(res, save_dir="/home/liyongbin/tmp_2"):
    _ = plt.figure(figsize=(15, 5))
    total_imgs = len(res['imgs'])
    for i in range(len(res['imgs'])):
        image = res['imgs'][i]
        seg_map = res['segs'][i]

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
        save_path = save_dir + "/" + str(i) + ".jpg"
        _.savefig(save_path)
        ax.cla()
        if (i+1) % 10 == 0:
            msg = "<<=====Finish Save Imgs: {}, {}/{} =======>>"\
                    .format(float(i+1)/float(total_imgs),
                    i+1, total_imgs)
            print(msg)

class DataSetClothes(DataSet):
    def __init__(self,dataset_file, mode):
        self.dataset_file = dataset_file
        self.mode = mode

    def prepare_image(self, annotation_path, image_dir, max_datasize=None):
        self.image_pathes = []
        items = [x.strip('\n') for x in open(annotation_path, "r")]
        items = items[1:]
        if max_datasize:
            items = items[:max_datasize]

        for item in items:
            image_path = item.split(",")[0]
            self.image_pathes.append(os.path.join(image_dir, image_path))
        self.datasize = len(self.image_pathes)
        self.index = 0

    def get_next_raw_image(self):
        image_filename = self.image_pathes[self.index]
        img = self.resize_image(Image.open(image_filename), True)
        self.index += 1
        return np.array([img])


def main(unused_argv):
    HOME_DIR = "/home/liyongbin/"
    HOME_IMG_DIR = "/home/liyongbin/"
    MODEL_DIR = \
            os.path.join(HOME_DIR,"models/research/deeplab/LIP/train")
    PB_DIR = \
        os.path.join(HOME_DIR,"models/research/deeplab/LIP/export/frozen_inference_graph.pb"
    save_res_dir = \
        os.path.join(HOME_DIR,"tmp_2")

    annotation_path = \
        os.path.join(HOME_IMG_DIR,"Mask_RCNN_BBox/train.csv")
    image_dir = \
        os.path.join(HOME_IMG_DIR,"Mask_RCNN_BBox/")

    model = DeeplabLIPModel()
    model.load_from_fpb_to_detect(pb_path=PB_DIR)

    dataset_test = DataSetClothes(
            dataset_file="",
            mode="inference")

    dataset_test.prepare_image(
            annotation_path=annotation_path,
            image_dir=image_dir,
            max_datasize=None
            )

    res = model.detect_img(dataset_test)
    save_res(res, save_res_dir)

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,3" 
    tf.app.run()

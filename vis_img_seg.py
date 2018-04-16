from matplotlib import gridspec
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import os

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
    print("Begin vis:", image.shape, seg_map.shape)
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

#    unique_labels = np.unique(seg_map)
#    ax = plt.subplot(grid_spec[3])
#    plt.imshow(
#      FULL_COLOR_MAP[unique_labels].astype(np.uint8), interpolation='nearest')
#    ax.yaxis.tick_right()
#    plt.yticks(range(len(unique_labels)), LABEL_NAMES[unique_labels])
#    plt.xticks([], [])
#    ax.tick_params(width=0.0)
    plt.grid('off')
    plt.show()

def get_img_res(filename="train_id.txt"):
    result_path = "/home/dalaoshe/tmp/"
    filenames = []
    for i in range(1, 40):
        name = "%06d" % i
        filenames.append(name)
    img_paths = [result_path + x + "_image.png" for x in filenames]
    seg_paths = [result_path + x + "_prediction.png" for x in filenames]
    return img_paths, seg_paths


if __name__ == "__main__":
    imgs,seg_paths = get_img_res("./val_id.txt")
    for i in range(len(imgs)):
        image_filename = \
        "/media/dalaoshe/D/LIP/TrainVal_images/TrainVal_images/val_images/3348_434303.jpg"
        #imgs[i]
        seg_filename = \
        "/media/dalaoshe/D/LIP/TrainVal_images/TrainVal_parsing_annotations/val_segmentations/3348_434303.png"

        if os.path.exists(image_filename) and\
                os.path.exists(seg_filename):
                    
            img = np.array(Image.open(image_filename))
            gt_seg = np.array(Image.open(seg_filename))
            vis_segmentation(img, gt_seg)
        else:
            print(i, image_filename, seg_filename)




     

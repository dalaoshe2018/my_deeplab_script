import sys
import os
from PIL import Image
import random
import numpy as np

class DataSet():
    INPUT_SIZE = 321
    def __init__(self, dataset, 
            mode="training",
            image_folder="",
            gt_seg_folder="",
            input_size=321):
        print('Processing ' + dataset)
        self.filenames = [x.strip('\n') for x in open(dataset, 'r')]
        self.filenames = self.filenames[:30000]
        self.mode = mode
        self.datasize = len(self.filenames)
        self.index = 0
        self.image_folder = image_folder
        self.gt_seg_folder = gt_seg_folder
        self.image_format = "jpg"
        self.seg_format = "png"
        self.INPUT_SIZE = input_size
        random.shuffle(self.filenames)

    def prepare_image(self):
        self.image_pathes = [os.path.join(
                self.image_folder, name\
                + '.' + self.image_format) for name in self.filenames]
        self.seg_pathes = [os.path.join(
                self.gt_seg_folder, name\
                + '.' + self.seg_format) for name in self.filenames]


    def resize_image(self, image, conv=True):
        width, height = image.size
        #resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        #target_size = (int(resize_ratio * width), int(resize_ratio * height))
        if max(width,height) < 300:
            target_size = [self.INPUT_SIZE, self.INPUT_SIZE]
        else:
            target_size = [self.INPUT_SIZE, self.INPUT_SIZE]

        if conv:
            resized_image = np.array(image.convert('RGB').\
                    resize(target_size, Image.BILINEAR))
        else:
            resized_image = np.array(image.resize(target_size, Image.NEAREST))
            resized_image = np.where(resized_image <= 19, resized_image, 0)
            


        #print("old size: {}, new size {}".format(image.size, resized_image.size))
        return resized_image

    def get_next_batch_datas(self, batch_size=1):
        if self.mode == "training":
            imgs = []
            gt_segs = []
            for i in range(batch_size):
                image_filename = os.path.join(
                self.image_folder, self.filenames[self.index]\
                        + '.' + self.image_format)
                seg_filename = os.path.join(
                self.gt_seg_folder, self.filenames[self.index]\
                        + '.' + self.seg_format)
                img = self.resize_image(Image.open(image_filename), True)
                #img = Image.open(image_filename)
                gt_seg = self.resize_image(Image.open(seg_filename), False)
                #gt_seg = Image.open(seg_filename)
                img_w,img_h = img.shape[:2]
                seg_w,seg_h = gt_seg.shape[:2]
                if img_w != seg_w or img_h != seg_h:
                    raise RuntimeError('Shape mismatched between \
                            image and label. %d %d %d %d'\
                            %(img_w,img_h,seg_w,seg_h))

                if len(np.array(img).shape) < 3:
                    raise RuntimeError('Image shape error\
                            image and label. {} {}'.\
                            format(np.array(img).shape, image_filename))


                imgs.append(img)
                gt_segs.append(gt_seg)


                self.index += 1
                if self.index >= self.datasize:
                    random.shuffle(self.filenames)
                    self.index = 0
                #print(np.shape(gt_segs), np.max(gt_segs))
            if np.max(gt_segs) >= 20:
                raise RuntimeError('Too max label {}'.\
                        format(np.max(gt_segs)))
            return imgs, gt_segs
    


        

if __name__ == "__main__":
    dataset = "/home/liyongbin/LIP/train_img/train_id.txt"
    img_folder = "/home/liyongbin/LIP/train_img/train_images"
    seg_folder = "/home/liyongbin/LIP/train_img/train_parse_annotation/train_segmentations"
    dataset_train = DataSet(dataset,
            image_folder=img_folder,
            gt_seg_folder=seg_folder)
    dataset_train.get_next_batch_datas()

from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil
from util.base_dataset import BaseDataset


class UPBDataset(BaseDataset):
    def __init__(self, *args, **kwargs):
        super(UPBDataset, self).__init__(*args, **kwargs)
        self.full_res_shape = (640, 360)

    def get_color(self, folder, frame_index, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color


class UPBRAWDataset(UPBDataset):
    def __init__(self, *args, **kwargs):
        super(UPBRAWDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index):
        image_path = os.path.join(self.data_path, folder, str(frame_index) + ".png")
        return image_path
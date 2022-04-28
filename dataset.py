################################################################################
################################ Python Imports ################################
################################################################################

import os
import cv2

from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

################################################################################
################################# Local Imports ################################
################################################################################


################################################################################
################################# Main Classes #################################
################################################################################

class OctDataset(Dataset):

    def __init__(self, image_dir, mask_dir, sort=False,
                 transforms=None, num_samples=-1):
      """
      image_dir -> src directory for image data
      label_dir -> src directory for label/maks dir
      sort -> arrange files in order according to their numeric postfix
      transforms -> transforms to be applied before presenting, training, etc
      num_sample -> how many total sample to read in (-1 default means all)
      """
      self.image_dir = image_dir
      self.mask_dir = mask_dir
      self.images = os.listdir(image_dir)

      if sort:
        self.images.sort()

      self.transforms = transforms

      if self.transforms is None:
            self.transforms = A.Compose([ToTensorV2()])

      if (num_samples >= 0):
            self.images = self.images[:num_samples]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        img_path  = os.path.join(
            self.image_dir, self.images[index])

        label_path = os.path.join(
            self.mask_dir, self.images[index].replace("image", "label"))

        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)

        augumentations = self.transforms(image=image, mask=mask)
        image, mask = augumentations['image'], augumentations['mask']

        return image, mask

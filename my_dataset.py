import random

from PIL import Image
import torch
from torch.utils.data import Dataset
from Augmentation import extract_and_stitch_grid_patches, extract_and_stitch_single_patches, extract_and_stitch_random_patches
from torchvision import transforms

class valDataSet(Dataset):
    """自定义数据集"""
    def __init__(self, images_path: list, images_class: list, transform1=None,transform2=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform1 = transform1
        self.transform2 = transform2

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        if img.mode != 'RGB':
            img = img.convert('RGB')

        img2 = img
        if self.transform1 and self.transform2 is not None:
            img1 = self.transform1(img)
            img2 = self.transform2(img)
        label = self.images_class[item]

        return img1, img2, label
    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, images1, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        images1 = torch.stack(images1, dim=0)
        labels = torch.as_tensor(labels)
        return images, images1, labels


class trainDataSet(Dataset):
    """自定义数据集"""
    def __init__(self, images_path: list, images_class: list, transform1=None, transform2=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform1 = transform1
        self.transform2 = transform2

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        if img.mode != 'RGB':
            img = img.convert('RGB')
        h = img.size[1]
        w = img.size[0]
        P = random.random()
        if P <= 0.2:
        #     # a = random.randint(h // 16, h // 4)
        #     # grid_size =[a,a]


            grid_size = [random.randint(4, 16), random.randint(4, 16)]
            # scale = random.choice([5, 4, 2, 1/0.75])
            scale = 1/0.5
            stride = random.choice([[x * scale for x in grid_size]])



        #     # if all(a == h // 32 for a in grid_size):
        #     #     mask_spacings = random.choice([[x * 1.5 for x in grid_size],
        #     #                                    [x * 2 for x in grid_size]]) #0.5
        #     # else:
        #     stride = random.choice([[x * 2 for x in grid_size]])


            mask_image_array, stitch_image_array = extract_and_stitch_grid_patches(img, grid_size, stride)
            mask_image = Image.fromarray(mask_image_array)
            stitch_image = Image.fromarray(stitch_image_array)
            if self.transform1 and self.transform2 is not None:
                mask_image = self.transform1(mask_image)
                stitch_image = self.transform2(stitch_image)

            # print(stitch_image.size(), mask_image.size())
            label = self.images_class[item]

        # if P <= 0.2:
        #     mask_image_array, stitch_image_array = extract_and_stitch_single_patches(img, patch_scale_size=0.75)
        #
        #     mask_image = Image.fromarray(mask_image_array)
        #     stitch_image = Image.fromarray(stitch_image_array)
        #     if self.transform1 and self.transform2 is not None:
        #         mask_image = self.transform1(mask_image)
        #         stitch_image = self.transform2(stitch_image)
        #
        #     label = self.images_class[item]

        # if P <= 0.2:
        #     mask_image_array, stitch_image_array = extract_and_stitch_random_patches(img,random.choice([4, 8, 16]), scale = 0.2)
        #
        #     mask_image = Image.fromarray(mask_image_array)
        #     stitch_image = Image.fromarray(stitch_image_array)
        #     if self.transform1 and self.transform2 is not None:
        #         mask_image = self.transform1(mask_image)
        #         stitch_image = self.transform2(stitch_image)
        #
        #     label = self.images_class[item]


        else:
            img1 = img
            if self.transform1 and self.transform2 is not None:
                stitch_aug = transforms.Compose([
                    # transforms.RandAugment(),
                    transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10)
                    # transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET)
                ])
                img1 = stitch_aug(img1)
                mask_image = self.transform1(img)
                stitch_image = self.transform2(img1)

            label = self.images_class[item]

        return mask_image, stitch_image, label

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, images1, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        images1 = torch.stack(images1, dim=0)
        labels = torch.as_tensor(labels)
        return images, images1, labels
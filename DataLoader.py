
import os
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import torch
import numpy as np
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import train_transforms, get_boxes_from_mask, init_point_sampling
import json
import random
from torch.nn.functional import interpolate

class TestingDataset(Dataset):
    
    def __init__(self, data_path, image_size=256, mode='test', requires_name=True, point_num=1, return_ori_mask=True, prompt_path=None):
        """
        Initializes a TestingDataset object.
        Args:
            data_path (str): The path to the data.
            image_size (int, optional): The size of the image. Defaults to 256.
            mode (str, optional): The mode of the dataset. Defaults to 'test'.
            requires_name (bool, optional): Indicates whether the dataset requires image names. Defaults to True.
            point_num (int, optional): The number of points to retrieve. Defaults to 1.
            return_ori_mask (bool, optional): Indicates whether to return the original mask. Defaults to True.
            prompt_path (str, optional): The path to the prompt file. Defaults to None.
        """
        self.image_size = image_size
        self.return_ori_mask = return_ori_mask
        self.prompt_path = prompt_path
        self.prompt_list = {} if prompt_path is None else json.load(open(prompt_path, "r"))
        self.requires_name = requires_name
        self.point_num = point_num

        json_file = open(os.path.join(data_path, f'Dynamic_{mode}.json'), "r")
        dataset = json.load(json_file)
    
        self.image_paths = list(dataset.values())
        self.label_paths = list(dataset.keys())
      
        self.pixel_mean = [123.675, 116.28, 103.53]
        self.pixel_std = [58.395, 57.12, 57.375]
    
    # def __getitem__(self, index):
    #     """
    #     Retrieves and preprocesses an item from the dataset.
    #     Args:
    #         index (int): The index of the item to retrieve.
    #     Returns:
    #         dict: A dictionary containing the preprocessed image and associated information.
    #     """
    #     image_input = {}
    #     try:
    #         image = cv2.imread(self.image_paths[index])
    #         image = (image - self.pixel_mean) / self.pixel_std
    #     except:
    #         print(self.image_paths[index])

    #     mask_path = self.label_paths[index]
    #     ori_np_mask = cv2.imread(mask_path, 0)
        
    #     if ori_np_mask.max() == 255:
    #         ori_np_mask = ori_np_mask / 255

    #     assert np.array_equal(ori_np_mask, ori_np_mask.astype(bool)), f"Mask should only contain binary values 0 and 1. {self.label_paths[index]}"

    #     h, w = ori_np_mask.shape
    #     ori_mask = torch.tensor(ori_np_mask).unsqueeze(0)

    #     transforms = train_transforms(self.image_size, h, w)
    #     augments = transforms(image=image, mask=ori_np_mask)
    #     image, mask = augments['image'], augments['mask'].to(torch.int64)

    #     if self.prompt_path is None:
    #         boxes = get_boxes_from_mask(mask, max_pixel = 0)
    #         point_coords, point_labels = init_point_sampling(mask, self.point_num)
    #     else:
    #         prompt_key = mask_path.split('/')[-1]
    #         boxes = torch.as_tensor(self.prompt_list[prompt_key]["boxes"], dtype=torch.float)
    #         point_coords = torch.as_tensor(self.prompt_list[prompt_key]["point_coords"], dtype=torch.float)
    #         point_labels = torch.as_tensor(self.prompt_list[prompt_key]["point_labels"], dtype=torch.int)

    #     image_input["image"] = image
    #     image_input["label"] = mask.unsqueeze(0)
    #     image_input["point_coords"] = point_coords
    #     image_input["point_labels"] = point_labels
    #     image_input["boxes"] = boxes
    #     image_input["original_size"] = (h, w)
    #     image_input["label_path"] = '/'.join(mask_path.split('/')[:-1])

    #     if self.return_ori_mask:
    #         image_input["ori_label"] = ori_mask
     
    #     image_name = self.label_paths[index].split('/')[-1]
    #     if self.requires_name:
    #         image_input["name"] = image_name
    #         return image_input
    #     else:
    #         return image_input


    def __getitem__(self, index):
        """
        Retrieves and preprocesses an item from the dataset.
        Args:
            index (int): The index of the item to retrieve.
        Returns:
            dict: A dictionary containing the preprocessed image and associated information.
        """
        image_input = {}
        try:
            # 读取图像并进行标准化
            image = cv2.imread(self.image_paths[index])
            image = (image - self.pixel_mean) / self.pixel_std
        except Exception as e:
            print(f"Error loading image {self.image_paths[index]}: {e}")

        # 读取对应的mask
        mask_path = self.label_paths[index]
        ori_np_mask = cv2.imread(mask_path, 0)

        # 归一化mask到二值
        if ori_np_mask.max() == 255:
            ori_np_mask = ori_np_mask / 255

        # 确保mask只有0和1
        assert np.array_equal(ori_np_mask, ori_np_mask.astype(bool)), f"Mask should only contain binary values 0 and 1. {self.label_paths[index]}"

        # 获取mask的尺寸
        h, w = ori_np_mask.shape

        # 检查图像和mask的尺寸是否一致
        if image.shape[:2] != (h, w):
            # print(f"Image and mask size mismatch: {image.shape[:2]} vs {ori_np_mask.shape}")
            # 如果尺寸不一致，调整图像尺寸以匹配mask的尺寸
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)

        # 将mask转换为tensor格式
        ori_mask = torch.tensor(ori_np_mask).unsqueeze(0)

        # 执行数据增强
        transforms = train_transforms(self.image_size, h, w)
        augments = transforms(image=image, mask=ori_np_mask)
        image, mask = augments['image'], augments['mask'].to(torch.int64)

        # 如果没有提供prompt路径，使用自动生成的提示
        if self.prompt_path is None:
            boxes = get_boxes_from_mask(mask, max_pixel=0)
            point_coords, point_labels = init_point_sampling(mask, self.point_num)
        else:
            # 如果提供了prompt路径，则从prompt文件中加载
            prompt_key = mask_path.split('/')[-1]
            boxes = torch.as_tensor(self.prompt_list[prompt_key]["boxes"], dtype=torch.float)
            point_coords = torch.as_tensor(self.prompt_list[prompt_key]["point_coords"], dtype=torch.float)
            point_labels = torch.as_tensor(self.prompt_list[prompt_key]["point_labels"], dtype=torch.int)

        # 将处理后的图像和mask及其他相关信息打包到字典中
        image_input["image"] = image
        image_input["label"] = mask.unsqueeze(0)
        image_input["point_coords"] = point_coords
        image_input["point_labels"] = point_labels
        image_input["boxes"] = boxes
        image_input["original_size"] = (h, w)
        image_input["label_path"] = '/'.join(mask_path.split('/')[:-1])

        # 如果需要返回原始的mask，则包含在返回值中
        if self.return_ori_mask:
            image_input["ori_label"] = ori_mask

        # 获取文件名并返回
        image_name = self.label_paths[index].split('/')[-1]
        if self.requires_name:
            image_input["name"] = image_name
            return image_input
        else:
            return image_input


    def __len__(self):
        return len(self.label_paths)


class TrainingDataset(Dataset):
    def __init__(self, data_dir, image_size=256, mode='train', requires_name=True, point_num=1, mask_num=5):
        """
        Initializes a training dataset.
        Args:
            data_dir (str): Directory containing the dataset.
            image_size (int, optional): Desired size for the input images. Defaults to 256.
            mode (str, optional): Mode of the dataset. Defaults to 'train'.
            requires_name (bool, optional): Indicates whether to include image names in the output. Defaults to True.
            num_points (int, optional): Number of points to sample. Defaults to 1.
            num_masks (int, optional): Number of masks to sample. Defaults to 5.
        """
        self.image_size = image_size
        self.requires_name = requires_name
        self.point_num = point_num
        self.mask_num = mask_num
        self.pixel_mean = [123.675, 116.28, 103.53]
        self.pixel_std = [58.395, 57.12, 57.375]

        dataset = json.load(open(os.path.join(data_dir, f'CAMUS_{mode}.json'), "r"))
        self.image_paths = list(dataset.keys())
        self.label_paths = list(dataset.values())
    # def __getitem__(self, index):
    #     """
    #     Returns a sample from the dataset.
    #     Args:
    #         index (int): Index of the sample.
    #     Returns:
    #         dict: A dictionary containing the sample data.
    #     """
    #     image_input = {}
    #     try:
    #         # 读取图像
    #         image = cv2.imread(self.image_paths[index])
    #         if image is None:
    #             raise FileNotFoundError(f"Unable to load image at {self.image_paths[index]}")

    #         # 标准化图像
    #         image = (image - self.pixel_mean) / self.pixel_std
    #     except:
    #         print(self.image_paths[index])

    #     h, w, _ = image.shape
    #     transforms = train_transforms(self.image_size, h, w)
        
    #     masks_list = []
    #     boxes_list = []
    #     point_coords_list, point_labels_list = [], []
    #     # labels_list = []  # 用于存储类别标签    add by yc

    #     # 随机选择mask路径
    #     mask_path = random.choices(self.label_paths[index], k=self.mask_num)
        
    #     for m in mask_path:
    #         pre_mask = cv2.imread(m, 0)  # 读取掩码

    #         # 归一化掩码
    #         if pre_mask.max() == 255:
    #             pre_mask = pre_mask / 255

    #         # 检查图像和掩码的尺寸是否一致
    #         if pre_mask.shape != image.shape[:2]:
    #             # print(f"Image and mask size mismatch: {image.shape[:2]} vs {pre_mask.shape}")
    #             pre_mask = cv2.resize(pre_mask, (w, h), interpolation=cv2.INTER_NEAREST)  # 调整掩码尺寸

    #         # 数据增强
    #         augments = transforms(image=image, mask=pre_mask)
    #         image_tensor, mask_tensor = augments['image'], augments['mask'].to(torch.int64)
           
    #        # 获取bounding box
    #         boxes = get_boxes_from_mask(mask_tensor)
    #         point_coords, point_label = init_point_sampling(mask_tensor, self.point_num)

    #         # 存储张量
    #         masks_list.append(mask_tensor)
    #         boxes_list.append(boxes)
    #         point_coords_list.append(point_coords)
    #         point_labels_list.append(point_label)
    #         # labels_list.append(class_label)

    #     # 堆叠mask、boxes和点的坐标和标签
    #     mask = torch.stack(masks_list, dim=0)
    #     boxes = torch.stack(boxes_list, dim=0)
    #     point_coords = torch.stack(point_coords_list, dim=0)
    #     point_labels = torch.stack(point_labels_list, dim=0)

    #     # 构建返回的字典
    #     image_input["image"] = image_tensor.unsqueeze(0)
    #     image_input["label"] = mask.unsqueeze(1)
    #     image_input["boxes"] = boxes
    #     image_input["point_coords"] = point_coords
    #     image_input["point_labels"] = point_labels
    #     # 获取图像名称
    #     image_name = self.image_paths[index].split('/')[-1]
        
    #     # 如果需要图像名称，则包含在返回值中
    #     if self.requires_name:
    #         image_input["name"] = image_name
    #         return image_input
    #     else:
    #         return image_input
    # def __len__(self):
    #     return len(self.image_paths)
    
    


    def __getitem__(self, index):
        """
        Returns a sample from the dataset.
        Args:
            index (int): Index of the sample.
        Returns:
            dict: A dictionary containing the sample data.
        """

        image_input = {}
        try:
            image = cv2.imread(self.image_paths[index])
            image = (image - self.pixel_mean) / self.pixel_std
        except:
            print(self.image_paths[index])

        h, w, _ = image.shape
        transforms = train_transforms(self.image_size, h, w)
    
        masks_list = []
        boxes_list = []
        point_coords_list, point_labels_list = [], []
        mask_path = random.choices(self.label_paths[index], k=self.mask_num)
        for m in mask_path:
            pre_mask = cv2.imread(m, 0)
            if pre_mask.max() == 255:
                pre_mask = pre_mask / 255

            # 检查图像和掩码的尺寸是否一致
            if pre_mask.shape != image.shape[:2]:
                # print(f"Image and mask size mismatch: {image.shape[:2]} vs {pre_mask.shape}")
                pre_mask = cv2.resize(pre_mask, (w, h), interpolation=cv2.INTER_NEAREST)  # 调整掩码尺寸

            augments = transforms(image=image, mask=pre_mask)
            image_tensor, mask_tensor = augments['image'], augments['mask'].to(torch.int64)

            boxes = get_boxes_from_mask(mask_tensor)
            point_coords, point_label = init_point_sampling(mask_tensor, self.point_num)

            masks_list.append(mask_tensor)
            boxes_list.append(boxes)
            point_coords_list.append(point_coords)
            point_labels_list.append(point_label)

        mask = torch.stack(masks_list, dim=0)
        boxes = torch.stack(boxes_list, dim=0)
        point_coords = torch.stack(point_coords_list, dim=0)
        point_labels = torch.stack(point_labels_list, dim=0)

        image_input["image"] = image_tensor.unsqueeze(0)
        image_input["label"] = mask.unsqueeze(1)
        image_input["boxes"] = boxes
        image_input["point_coords"] = point_coords
        image_input["point_labels"] = point_labels

        image_name = self.image_paths[index].split('/')[-1]
        if self.requires_name:
            image_input["name"] = image_name
            return image_input
        else:
            return image_input
    def __len__(self):
        return len(self.image_paths)


# class TrainingDataset(Dataset):
#     def __init__(self, data_dir, image_size=256, mode='train', requires_name=True, point_num=1, mask_num=5):
#         """
#         Initializes a training dataset.
#         Args:
#             data_dir (str): Directory containing the dataset.
#             image_size (int, optional): Desired size for the input images. Defaults to 256.
#             mode (str, optional): Mode of the dataset. Defaults to 'train'.
#             requires_name (bool, optional): Indicates whether to include image names in the output. Defaults to True.
#             num_points (int, optional): Number of points to sample. Defaults to 1.
#             num_masks (int, optional): Number of masks to sample. Defaults to 5.
#         """
#         self.image_size = image_size
#         self.requires_name = requires_name
#         self.point_num = point_num
#         self.mask_num = mask_num
#         self.pixel_mean = [123.675, 116.28, 103.53]
#         self.pixel_std = [58.395, 57.12, 57.375]

#         dataset = json.load(open(os.path.join(data_dir, f'Cradiac_{mode}.json'), "r"))
#         self.image_paths = list(dataset.keys())
#         self.label_paths = list(dataset.values())
        

#     def __getitem__(self, index):
#         """
#         Returns a sample from the dataset.
#         Args:
#             index (int): Index of the sample.
#         Returns:
#             dict: A dictionary containing the sample data.
#         """
#         image_input = {}
#         try:
#             # 读取图像
#             image = cv2.imread(self.image_paths[index])
#             if image is None:
#                 raise FileNotFoundError(f"Unable to load image at {self.image_paths[index]}")

#             # 标准化图像
#             image = (image - self.pixel_mean) / self.pixel_std
#         except:
#             print(self.image_paths[index])

#         h, w, _ = image.shape
#         transforms = train_transforms(self.image_size, h, w)
        
#         masks_list = []
#         boxes_list = []
#         point_coords_list, point_labels_list = [], []


#         # 随机选择mask路径
#         mask_path = random.choices(self.label_paths[index], k=self.mask_num)
        
#         for m in mask_path:
#             pre_mask = cv2.imread(m, 0)  # 读取掩码
#             # 归一化掩码
#             if pre_mask.max() == 255:
#                 pre_mask = pre_mask / 255

#             # 检查图像和掩码的尺寸是否一致
#             if pre_mask.shape != image.shape[:2]:
#                 # print(f"Image and mask size mismatch: {image.shape[:2]} vs {pre_mask.shape}")
#                 pre_mask = cv2.resize(pre_mask, (w, h), interpolation=cv2.INTER_NEAREST)  # 调整掩码尺寸

#             # 数据增强
#             augments = transforms(image=image, mask=pre_mask)
#             image_tensor, mask_tensor = augments['image'], augments['mask'].to(torch.int64)

#             # 获取bounding box
#             boxes = get_boxes_from_mask(mask_tensor)
#             point_coords, point_label = init_point_sampling(mask_tensor, self.point_num)

#             # 存储张量
#             masks_list.append(mask_tensor)
#             boxes_list.append(boxes)
#             point_coords_list.append(point_coords)
#             point_labels_list.append(point_label)

#         # 堆叠mask、boxes和点的坐标和标签
#         mask = torch.stack(masks_list, dim=0)
#         boxes = torch.stack(boxes_list, dim=0)
#         point_coords = torch.stack(point_coords_list, dim=0)
#         point_labels = torch.stack(point_labels_list, dim=0)

#         # 构建返回的字典
#         image_input["image"] = image_tensor.unsqueeze(0)
#         image_input["label"] = mask.unsqueeze(1)
#         image_input["boxes"] = boxes
#         image_input["point_coords"] = point_coords
#         image_input["point_labels"] = point_labels

#         # 获取图像名称
#         image_name = self.image_paths[index].split('/')[-1]
        
#         # 如果需要图像名称，则包含在返回值中
#         if self.requires_name:
#             image_input["name"] = image_name
#             return image_input
#         else:
#             return image_input



#     def __len__(self):
#         return len(self.image_paths)




def stack_dict_batched(batched_input):
    out_dict = {}
    for k,v in batched_input.items():
        if isinstance(v, list):
            out_dict[k] = v
        else:
            out_dict[k] = v.reshape(-1, *v.shape[2:])
    return out_dict


if __name__ == "__main__":
    train_dataset = TrainingDataset("/media/Storage4/yc/SAM-Med2D/Head", image_size=256, mode='train', requires_name=True, point_num=1, mask_num=18)
    print("Dataset:", len(train_dataset))
    train_batch_sampler = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True, num_workers=4)
    for i, batched_image in enumerate(tqdm(train_batch_sampler)):
        batched_image = stack_dict_batched(batched_image)
        print(batched_image["image"].shape, batched_image["label"].shape)


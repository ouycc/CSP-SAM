from albumentations.pytorch import ToTensorV2
import cv2
import albumentations as A
import torch
import numpy as np
from torch.nn import functional as F
from skimage.measure import label, regionprops
from matplotlib import pyplot as plt
import random
import torch.nn as nn
import logging
import os
from torch.distributions.dirichlet import Dirichlet
import kornia
from segment_anything.utils.transforms import ResizeLongestSide
from refine import prepare_image, extract_bboxes_expand, extract_points, extract_mask



def sam_input_prepare(patch_input, pred_masks, image_embeddings=None, resize_transform=None, use_point=True, use_box=True, use_mask=True, add_neg=True, margin=0.0, gamma=1.0, strength=15):
    ori_size = pred_masks.shape[-2:]
    # input_dict = {
    #      'image': image,
    #      'original_size': ori_size,
    #      }
    image=patch_input["image"]
    image= image.repeat(5, 1, 1, 1)
    target_size = (128,128)
    expand_list = torch.zeros((len(pred_masks))).to(image.device)
    if use_box:
        bboxes, box_masks, areas, expand_list = extract_bboxes_expand(image_embeddings, pred_masks, margin=margin)
        patch_input['boxes'] = bboxes
    else:
        patch_input['boxes'] = None
    
    point_coords, point_labels, gaus_dt = extract_points(pred_masks, add_neg=add_neg, use_mask=use_mask, gamma=gamma)
    if use_point:
        patch_input['point_coords'] = point_coords
        patch_input['point_labels'] = point_labels
    else:
        # 给默认值以防止后续 prompt_encoder 报错
        B = pred_masks.shape[0]  # batch size
        patch_input['point_coords'] = torch.zeros(B, 1, 2).to(pred_masks.device)
        patch_input['point_labels'] = torch.ones(B, 1).to(pred_masks.device)
        
    
        
    if use_mask:
        patch_input['mask_inputs'] = extract_mask(pred_masks, gaus_dt, target_size, is01=True, strength=strength, device=image.device, expand_list=expand_list)
    else:
        B = pred_masks.shape[0]
        H, W = target_size
        patch_input['mask_inputs'] = torch.zeros(B, 1, H, W).to(pred_masks.device)
   
    return patch_input,point_coords


def sam_refiner(batch_input, 
                coarse_masks,
                sam,
                image_embeddings,
                cnn_feature_list,
                resize_transform=None,
                use_point=True,
                use_box=True,
                use_mask=True,
                add_neg=True,
                iters=1,
                margin=0.0,
                gamma=4.0,
                strength=30,
                use_samhq=False,
                ddp=False,
                is_train=True):
    """
    SAMRefiner refines coarse masks from an image by generating noise-tolerant prompts for SAM.

    Arguments:
      image_path (str): The image path for the target image.
      coarse_masks (list(array) or array): The coarse masks to be refined.
      sam (Sam): The Sam model.
      resize_transform (list(float)): The resize_transform used in sam. Default: ResizeLongestSide.
      use_point (bool): Whether to use point prompts. Default: True
      use_box (bool): Whether to use box prompts. Default: True
      use_mask (bool): Whether to use mask prompts. Default: True
      add_neg (bool): Whether to use the negative point prompts. Default: True
      iters (int): The number of iterative refinement. Default: 5
      margin (float): The parameter used to control whether to enlarge the box. Default: 0 (not enlarge)
      gamma (float): The parameter used to control the span of Gaussian distribution in mask prompt. Default: 4.0
      gamma (float): The parameter used to control the amplitude of Gaussian distribution in mask prompt. Default: 30
      use_samhq (bool): Whether to use samhq model. Default: False
    """
    
    # if isinstance(coarse_masks, list):
    #     coarse_masks = np.stack(coarse_masks, axis=0)
        
    # if len(coarse_masks.shape) == 2:
    #     coarse_masks = coarse_masks[None: ,]
    # coarse_masks = torch.tensor(coarse_masks, dtype=torch.uint8).to(sam.device)
        
    # assert len(coarse_masks.shape) == 3, "coarse mask dim must be (n, h, w), but got {}".format(coarse_masks.shape)

    # if resize_transform is None:
    #     resize_transform = ResizeLongestSide(sam.image_encoder.img_size)
    
    # image = cv2.imread(image_path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # ori_size = image.shape[:2]
    # image = [prepare_image(image, resize_transform, sam.device)]
    
    # with torch.no_grad():
    #     if ddp:
    #         input_images = torch.stack([sam.module.preprocess(x) for x in image], dim=0)
    #         if not use_samhq:
    #             image_embeddings = sam.module.image_encoder(input_images) # torch.Size([1, 256, 64, 64])
    #         else:
    #             image_embeddings, interm_embeddings = sam.module.image_encoder(input_images)
    #             interm_embeddings = interm_embeddings[0] # early layer
    #     else:
    #         input_images = torch.stack([sam.preprocess(x) for x in image], dim=0)
    #         if not use_samhq:
    #             image_embeddings = sam.image_encoder(input_images) # torch.Size([1, 256, 64, 64])
    #         else:
    #             image_embeddings, interm_embeddings = sam.image_encoder(input_images)
    #             interm_embeddings = interm_embeddings[0] # early layer
    
    for i in range(iters):
        if i == 0:
            pred_mask_list = coarse_masks
        else:
            pred_mask_list = sam_masks_list.to(torch.uint8)
        
        input_dict, point_coords = sam_input_prepare(batch_input,
                                                     pred_mask_list,
                                                     image_embeddings,
                                                     resize_transform,
                                                     use_point=use_point,
                                                     use_box=use_box,
                                                     use_mask=use_mask,
                                                     add_neg=add_neg,
                                                     margin=margin,
                                                     gamma=gamma,
                                                     strength=strength)
        
        sam_input = [input_dict]
        
        if not is_train:
            with torch.no_grad():
                if ddp:
                    if not use_samhq:
                        sam_output = sam.module.forward_with_image_embeddings(image_embeddings, sam_input, multimask_output=True)[0] #dict_keys(['masks', 'iou_predictions', 'low_res_logits'])
                    # else:
                    #     sam_output = sam.module.forward_with_image_embeddings(image_embeddings, interm_embeddings,sam_input, multimask_output=True)[0] #dict_keys(['masks', 'iou_predictions', 'low_res_logits'])
                else:
                    if not use_samhq:
                        # sam_output = sam.forward_with_image_embeddings(image_embeddings, sam_input, multimask_output=True)[0]
                        sam_output = sam.forward_with_image_embeddings(image_embeddings,cnn_feature_list, sam_input, multimask_output=True)[0] #dict_keys(['masks', 'iou_predictions', 'low_res_logits'])
                    # else:
                    #     sam_output = sam.forward_with_image_embeddings(image_embeddings, interm_embeddings,sam_input, multimask_output=True)[0] #dict_keys(['masks', 'iou_predictions', 'low_res_logits'])
        else:
            if ddp:
                sam_output = sam.module.forward_with_image_embeddings(image_embeddings, sam_input, multimask_output=True)[0] #dict_keys(['masks', 'iou_predictions', 'low_res_logits'])
            else:
                sam_output = sam.forward_with_image_embeddings(image_embeddings,cnn_feature_list, sam_input, multimask_output=True)[0]
                # sam_output = sam.forward_with_image_embeddings(image_embeddings, sam_input, multimask_output=True)[0] #dict_keys(['masks', 'iou_predictions', 'low_res_logits'])

        sam_masks = sam_output['masks']
        sam_masks3 = sam_masks.clone().detach()
        sam_ious = sam_output['iou_predictions']
        sam_masks_logits = sam_output["low_res_logits"]

        if is_train:
            return sam_masks, sam_ious, sam_masks3
        sam_masks_list = []
        sam_masks_logits_list = []
        cnt = 0
        for sm, si in zip(sam_masks, sam_ious):
            max_idx = torch.argmax(si)
            sam_masks_list.append(sm[max_idx])
            sam_masks_logits_list.append(sam_masks_logits[cnt][max_idx])
            cnt += 1

        sam_masks = torch.stack(sam_masks_list, dim=0)
        sam_masks=sam_masks.unsqueeze(0)
        sam_masks_logits = torch.stack(sam_masks_logits_list, dim=0)

        sam_masks_list = sam_masks > 0
        
    refined_masks = sam_masks_list.cpu().numpy().astype(np.uint8)
    assert len(refined_masks) == len(coarse_masks)
    return refined_masks, sam_ious, sam_masks3







class MLFusion(nn.Module):
    def __init__(self, norm, act):
        super().__init__()
        self.fusi_conv = nn.Sequential(
            nn.Conv2d(1024, 256, 1,bias = False),
            norm(256),
            act(),
        )

        self.attn_conv = nn.ModuleList()
        for i in range(4):
            self.attn_conv.append(nn.Sequential(
                nn.Conv2d(256, 256, 1,bias = False),
                norm(256),
                act(),
            ))

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, feature_list):
        device = feature_list[-1].device
        self.fusi_conv=self.fusi_conv.to(device)
        self.attn_conv=self.attn_conv.to(device)
        self.pool=self.pool.to(device)
        self.sigmoid=self.sigmoid.to(device)
        fusi_feature = torch.cat(feature_list, dim = 1).contiguous()
        fusi_feature = self.fusi_conv(fusi_feature)

        for i in range(4):
            x = feature_list[i]
            attn = self.attn_conv[i](x)
            attn = self.pool(attn)
            attn = self.sigmoid(attn)

            x = attn * x + x
            feature_list[i] = x
        
        return feature_list[0] + feature_list[1] + feature_list[2] + feature_list[3]

def LossFunc(pred, mask):
    mask=mask.float()
    bce = F.binary_cross_entropy(pred, mask, reduce=None)

    inter = ((pred * mask)).sum(dim=(2, 3))
    union = ((pred + mask)).sum(dim=(2, 3))
    aiou = 1 - (inter + 1) / (union - inter + 1)

    mae = F.l1_loss(pred, mask, reduce=None)

    return (bce + aiou + mae).mean()




def get_deep_feature_conv(norm, act):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        nn.Conv2d(256, 64, 3, padding=1, bias=False),
        norm(64),
        act(),
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        nn.Conv2d(64, 32, 3, padding=1, bias=False),
        norm(32),
        act(),
    )


class MEEM(nn.Module):
    def __init__(self, in_dim, hidden_dim, width, norm, act):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.width = width
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, 1, bias = False),
            norm(hidden_dim),
            nn.Sigmoid()
        )

        self.pool = nn.AvgPool2d(3, stride= 1,padding = 1)

        self.mid_conv = nn.ModuleList()
        self.edge_enhance = nn.ModuleList()
        for i in range(width - 1):
            self.mid_conv.append(nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 1, bias = False),
                norm(hidden_dim),
                nn.Sigmoid()
            ))
            self.edge_enhance.append(EdgeEnhancer(hidden_dim, norm, act))

        self.out_conv = nn.Sequential(
            nn.Conv2d(hidden_dim * width, in_dim, 1, bias = False),
            norm(in_dim),
            act()
        )
    
    def forward(self, x):
        mid = self.in_conv(x)

        out = mid
        #print(out.shape)
        
        for i in range(self.width - 1):
            mid = self.pool(mid)
            mid = self.mid_conv[i](mid)

            out = torch.cat([out, self.edge_enhance[i](mid)], dim = 1)
        
        out = self.out_conv(out)

        return out

class EdgeEnhancer(nn.Module):
    def __init__(self, in_dim, norm, act):
        super().__init__()
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 1, bias = False),
            norm(in_dim),
            nn.Sigmoid()
        )
        self.pool = nn.AvgPool2d(3, stride= 1, padding = 1)
    
    def forward(self, x):
        edge = self.pool(x)
        edge = x - edge
        edge = self.out_conv(edge)
        return x + edge
    
class DetailEnhancement(nn.Module):
    def __init__(self, img_dim, feature_dim, norm, act):
        super().__init__()

        self.img_in_conv = nn.Sequential(
            nn.Conv2d(3, img_dim, 3, padding = 1, bias = False),
            norm(img_dim),
            act()
        )
        self.img_er = MEEM(img_dim, img_dim  // 2, 4, norm, act)

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(feature_dim + img_dim, 32, 3, padding = 1, bias = False),
            norm(32),
            act(),
            nn.Conv2d(32, 16, 3, padding = 1, bias = False),
            norm(16),
            act(),
        )

        self.out_conv = nn.Conv2d(16, 1, 1)
        
        self.feature_upsample = nn.Sequential(
            nn.Conv2d(feature_dim * 2, feature_dim, 3, padding = 1, bias = False),
            norm(feature_dim),
            act(),
            nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = False),
            nn.Conv2d(feature_dim, feature_dim, 3, padding = 1, bias = False),
            norm(feature_dim),
            act(),
            # nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = False),
            # nn.Conv2d(feature_dim, feature_dim, 3, padding = 1, bias = False),
            # norm(feature_dim),
            # act(),
        )
    
    def forward(self, img, feature, b_feature):

        device = feature.device
        self.feature_upsample=self.feature_upsample.to(device) 
        self.img_er = self.img_er.to(device)  
        self.fusion_conv = self.fusion_conv.to(device)  
        self.out_conv = self.out_conv.to(device)
        self.img_in_conv=self.img_in_conv.to(device)
        feature = torch.cat([feature, b_feature], dim = 1)
        feature = self.feature_upsample(feature)

        img_feature = self.img_in_conv(img)
        img_feature = self.img_er(img_feature) + img_feature

        # img_feature = img_feature.repeat(5, 1, 1, 1)  # 在 batch 维度 (dim=0) 上重复 5 次
        out_feature = torch.cat([feature, img_feature], dim = 1)
        out_feature = self.fusion_conv(out_feature)
        out = self.out_conv(out_feature)

        return out


def generate_uncertain_point(masks, labels, low_res_masks, batched_input, point_num,patch_size=16):
    """
    生成不确定性点，改进：
    1. **高斯滤波**降低 SAM 对边缘的敏感性
    2. **Patch-based 选择**，避免点过度集中在边界
    """
    masks_clone = masks.clone()

    # === 1. 高斯滤波 + 梯度约束（减少边缘敏感性） ===
    masks_clone = kornia.filters.gaussian_blur2d(masks_clone, (3, 3), (1.0, 1.0))

    sobel_x = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]]).float().to(masks.device)
    sobel_y = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]]).float().to(masks.device)

    grad_x = F.conv2d(masks_clone, sobel_x, padding=1)
    grad_y = F.conv2d(masks_clone, sobel_y, padding=1)
    gradient_mag = torch.sqrt(grad_x**2 + grad_y**2)

    masks_clone = masks_clone - 0.1 * gradient_mag  # 降低边缘影响

    # === 2. 计算 Dirichlet 证据 ===
    evidence = F.softplus(masks_clone)
    alpha = evidence + 1
    S = torch.sum(alpha, dim=1, keepdim=True)

    # === 3. 计算熵不确定性 ===
    p = alpha / S
    uncertainty = -torch.sum(p * torch.log(p + 1e-6), dim=1, keepdim=True)

    # 计算二分类 mask
    masks_binary = (torch.sigmoid(alpha) > 0.5).float()

    # 低分辨率 mask
    low_res_masks_logist = torch.sigmoid(low_res_masks.clone())

    # === 4. Patch-based 选择高不确定性点 ===
    points, point_labels = patch_based_uncertain_points(masks_binary, labels, uncertainty, point_num, patch_size)

    # 更新 batched_input
    batched_input.update({
        "mask_inputs": low_res_masks_logist,
        "point_coords": torch.as_tensor(points),
        "point_labels": torch.as_tensor(point_labels),
        "boxes": None
    })

    return batched_input

def patch_based_uncertain_points(pred, gt, uncertainty, point_num=9, patch_size=16):
    """
    结合全局高不确定性区域与局部 Patch 进行点选
    """
    pred, gt = pred.cpu().numpy(), gt.cpu().numpy()
    uncertainty = uncertainty.cpu().detach().numpy()

    batch_points, batch_labels = [], []

    for j in range(pred.shape[0]):  
        one_pred = pred[j].squeeze(0)
        one_gt = gt[j].squeeze(0)
        one_uncertainty = uncertainty[j].squeeze(0)

        # === 1. 全局选择高不确定性区域 ===
        flat_indices = np.argsort(one_uncertainty.ravel())[::-1]  # 降序
        top_uncertainty_indices = np.unravel_index(flat_indices[:point_num * 3], one_uncertainty.shape)

        # === 2. 随机选择 Patch ===
        patch_half = patch_size // 2
        selected_points = []
        selected_labels = []

        for _ in range(point_num):
            idx = np.random.randint(0, len(top_uncertainty_indices[0]))  # 随机选取高不确定性点
            y, x = top_uncertainty_indices[0][idx], top_uncertainty_indices[1][idx]

            # 在 patch 范围内随机选择
            y_patch = np.clip(y + np.random.randint(-patch_half, patch_half + 1), 0, one_uncertainty.shape[0] - 1)
            x_patch = np.clip(x + np.random.randint(-patch_half, patch_half + 1), 0, one_uncertainty.shape[1] - 1)

            # 获取该点的类别
            label = 1 if one_pred[y_patch, x_patch] == 0 and one_gt[y_patch, x_patch] == 1 else 0

            selected_points.append((x_patch, y_patch))
            selected_labels.append(label)

        batch_points.append(selected_points)
        batch_labels.append(selected_labels)

    return np.array(batch_points), np.array(batch_labels)




def get_boxes_from_mask(mask, box_num=1, std = 0.1, max_pixel = 5):
    """
    Args:
        mask: Mask, can be a torch.Tensor or a numpy array of binary mask.
        box_num: Number of bounding boxes, default is 1.
        std: Standard deviation of the noise, default is 0.1.
        max_pixel: Maximum noise pixel value, default is 5.
    Returns:
        noise_boxes: Bounding boxes after noise perturbation, returned as a torch.Tensor.
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.numpy()
        
    label_img = label(mask)
    regions = regionprops(label_img)

    # Iterate through all regions and get the bounding box coordinates
    boxes = [tuple(region.bbox) for region in regions]


    # Handle the case when no boxes are detected
    if len(boxes) == 0:
        # Return a default box or an empty tensor based on your requirement
        return torch.zeros((box_num, 4), dtype=torch.float)

    # If the generated number of boxes is greater than the number of categories,
    # sort them by region area and select the top n regions
    if len(boxes) >= box_num:
        sorted_regions = sorted(regions, key=lambda x: x.area, reverse=True)[:box_num]
        boxes = [tuple(region.bbox) for region in sorted_regions]

    # If the generated number of boxes is less than the number of categories,
    # duplicate the existing boxes
    elif len(boxes) < box_num:
        num_duplicates = box_num - len(boxes)
        boxes += [boxes[i % len(boxes)] for i in range(num_duplicates)]

    # Perturb each bounding box with noise
    noise_boxes = []
    for box in boxes:
        y0, x0,  y1, x1  = box
        width, height = abs(x1 - x0), abs(y1 - y0)
        # Calculate the standard deviation and maximum noise value
        noise_std = min(width, height) * std
        max_noise = min(max_pixel, int(noise_std * 5))
         # Add random noise to each coordinate
        try:
            noise_x = np.random.randint(-max_noise, max_noise)
        except:
            noise_x = 0
        try:
            noise_y = np.random.randint(-max_noise, max_noise)
        except:
            noise_y = 0
        x0, y0 = x0 + noise_x, y0 + noise_y
        x1, y1 = x1 + noise_x, y1 + noise_y
        noise_boxes.append((x0, y0, x1, y1))
    return torch.as_tensor(noise_boxes, dtype=torch.float)


def select_random_points(pr, gt, point_num = 9):
    """
    Selects random points from the predicted and ground truth masks and assigns labels to them.
    Args:
        pred (torch.Tensor): Predicted mask tensor.
        gt (torch.Tensor): Ground truth mask tensor.
        point_num (int): Number of random points to select. Default is 9.
    Returns:
        batch_points (np.array): Array of selected points coordinates (x, y) for each batch.
        batch_labels (np.array): Array of corresponding labels (0 for background, 1 for foreground) for each batch.
    """
    pred, gt = pr.data.cpu().numpy(), gt.data.cpu().numpy()
    error = np.zeros_like(pred)
    error[pred != gt] = 1

    # error = np.logical_xor(pred, gt)
    batch_points = []
    batch_labels = []
    for j in range(error.shape[0]):
        one_pred = pred[j].squeeze(0)
        one_gt = gt[j].squeeze(0)
        one_erroer = error[j].squeeze(0)

        indices = np.argwhere(one_erroer == 1)
        if indices.shape[0] > 0:
            selected_indices = indices[np.random.choice(indices.shape[0], point_num, replace=True)]
        else:
            indices = np.random.randint(0, 256, size=(point_num, 2))
            selected_indices = indices[np.random.choice(indices.shape[0], point_num, replace=True)]
        selected_indices = selected_indices.reshape(-1, 2)

        points, labels = [], []
        for i in selected_indices:
            x, y = i[0], i[1]
            if one_pred[x,y] == 0 and one_gt[x,y] == 1:
                label = 1
            elif one_pred[x,y] == 1 and one_gt[x,y] == 0:
                label = 0
            else:
                label = -1
            points.append((y, x))   #Negate the coordinates
            labels.append(label)

        batch_points.append(points)
        batch_labels.append(labels)
    return np.array(batch_points), np.array(batch_labels)


def init_point_sampling(mask, get_point=1):
    """
    Initialization samples points from the mask and assigns labels to them.
    Args:
        mask (torch.Tensor): Input mask tensor.
        num_points (int): Number of points to sample. Default is 1.
    Returns:
        coords (torch.Tensor): Tensor containing the sampled points' coordinates (x, y).
        labels (torch.Tensor): Tensor containing the corresponding labels (0 for background, 1 for foreground).
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.numpy()
        
     # Get coordinates of black/white pixels
    fg_coords = np.argwhere(mask == 1)[:,::-1]
    bg_coords = np.argwhere(mask == 0)[:,::-1]

    fg_size = len(fg_coords)
    bg_size = len(bg_coords)

    if get_point == 1:
        if fg_size > 0:
            index = np.random.randint(fg_size)
            fg_coord = fg_coords[index]
            label = 1
        else:
            index = np.random.randint(bg_size)
            fg_coord = bg_coords[index]
            label = 0
        return torch.as_tensor([fg_coord.tolist()], dtype=torch.float), torch.as_tensor([label], dtype=torch.int)
    else:
        num_fg = get_point // 2
        num_bg = get_point - num_fg
        fg_indices = np.random.choice(fg_size, size=num_fg, replace=True)
        bg_indices = np.random.choice(bg_size, size=num_bg, replace=True)
        fg_coords = fg_coords[fg_indices]
        bg_coords = bg_coords[bg_indices]
        coords = np.concatenate([fg_coords, bg_coords], axis=0)
        labels = np.concatenate([np.ones(num_fg), np.zeros(num_bg)]).astype(int)
        indices = np.random.permutation(get_point)
        coords, labels = torch.as_tensor(coords[indices], dtype=torch.float), torch.as_tensor(labels[indices], dtype=torch.int)
        return coords, labels
    

def train_transforms(img_size, ori_h, ori_w):
    transforms = []
    if ori_h < img_size and ori_w < img_size:
        transforms.append(A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0)))
    else:
        transforms.append(A.Resize(int(img_size), int(img_size), interpolation=cv2.INTER_NEAREST))
    transforms.append(ToTensorV2(p=1.0))
    return A.Compose(transforms, p=1.)


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def generate_point(masks, labels, low_res_masks, batched_input, point_num):
    masks_clone = masks.clone()
    masks_sigmoid = torch.sigmoid(masks_clone)
    masks_binary = (masks_sigmoid > 0.5).float()

    low_res_masks_clone = low_res_masks.clone()
    low_res_masks_logist = torch.sigmoid(low_res_masks_clone)

    points, point_labels = select_random_points(masks_binary, labels, point_num = point_num)
    batched_input["mask_inputs"] = low_res_masks_logist
    batched_input["point_coords"] = torch.as_tensor(points)
    batched_input["point_labels"] = torch.as_tensor(point_labels)
    batched_input["boxes"] = None
    return batched_input


def setting_prompt_none(batched_input):
    batched_input["point_coords"] = None
    batched_input["point_labels"] = None
    batched_input["boxes"] = None
    return batched_input


def draw_boxes(img, boxes):
    img_copy = np.copy(img)
    for box in boxes:
        cv2.rectangle(img_copy, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    return img_copy


def save_masks(preds, save_path, mask_name, image_size, original_size, pad=None,  boxes=None, points=None, visual_prompt=False):
    ori_h, ori_w = original_size

    preds = torch.sigmoid(preds)
    preds[preds > 0.5] = int(1)
    preds[preds <= 0.5] = int(0)

    mask = preds.squeeze().cpu().numpy()
    mask = cv2.cvtColor(mask * 255, cv2.COLOR_GRAY2BGR)

    if visual_prompt: #visualize the prompt
        if boxes is not None:
            boxes = boxes.squeeze().cpu().numpy()

            x0, y0, x1, y1 = boxes
            if pad is not None:
                x0_ori = int((x0 - pad[1]) + 0.5)
                y0_ori = int((y0 - pad[0]) + 0.5)
                x1_ori = int((x1 - pad[1]) + 0.5)
                y1_ori = int((y1 - pad[0]) + 0.5)
            else:
                x0_ori = int(x0 * ori_w / image_size) 
                y0_ori = int(y0 * ori_h / image_size) 
                x1_ori = int(x1 * ori_w / image_size) 
                y1_ori = int(y1 * ori_h / image_size)

            boxes = [(x0_ori, y0_ori, x1_ori, y1_ori)]
            mask = draw_boxes(mask, boxes)

        if points is not None:
            point_coords, point_labels = points[0].squeeze(0).cpu().numpy(),  points[1].squeeze(0).cpu().numpy()
            point_coords = point_coords.tolist()
            if pad is not None:
                ori_points = [[int((x * ori_w / image_size)) , int((y * ori_h / image_size))]if l==0 else [x - pad[1], y - pad[0]]  for (x, y), l in zip(point_coords, point_labels)]
            else:
                ori_points = [[int((x * ori_w / image_size)) , int((y * ori_h / image_size))] for x, y in point_coords]

            for point, label in zip(ori_points, point_labels):
                x, y = map(int, point)
                color = (0, 255, 0) if label == 1 else (0, 0, 255)
                mask[y, x] = color
                cv2.drawMarker(mask, (x, y), color, markerType=cv2.MARKER_CROSS , markerSize=7, thickness=2)  
    os.makedirs(save_path, exist_ok=True)
    mask_path = os.path.join(save_path, f"{mask_name}")
    cv2.imwrite(mask_path, np.uint8(mask))


#Loss funcation
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred, mask):
        """
        pred: [B, 1, H, W]
        mask: [B, 1, H, W]
        """
        assert pred.shape == mask.shape, "pred and mask should have the same shape."
        p = torch.sigmoid(pred)
        num_pos = torch.sum(mask)
        num_neg = mask.numel() - num_pos
        w_pos = (1 - p) ** self.gamma
        w_neg = p ** self.gamma

        loss_pos = -self.alpha * mask * w_pos * torch.log(p + 1e-12)
        loss_neg = -(1 - self.alpha) * (1 - mask) * w_neg * torch.log(1 - p + 1e-12)

        loss = (torch.sum(loss_pos) + torch.sum(loss_neg)) / (num_pos + num_neg + 1e-12)

        return loss


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, mask):
        """
        pred: [B, 1, H, W]
        mask: [B, 1, H, W]
        """
        assert pred.shape == mask.shape, "pred and mask should have the same shape."
        p = torch.sigmoid(pred)
        intersection = torch.sum(p * mask)
        union = torch.sum(p) + torch.sum(mask)
        dice_loss = (2.0 * intersection + self.smooth) / (union + self.smooth)

        return 1 - dice_loss


class MaskIoULoss(nn.Module):

    def __init__(self, ):
        super(MaskIoULoss, self).__init__()

    def forward(self, pred_mask, ground_truth_mask, pred_iou):
        """
        pred_mask: [B, 1, H, W]
        ground_truth_mask: [B, 1, H, W]
        pred_iou: [B, 1]
        """
        assert pred_mask.shape == ground_truth_mask.shape, "pred_mask and ground_truth_mask should have the same shape."

        p = torch.sigmoid(pred_mask)
        intersection = torch.sum(p * ground_truth_mask)
        union = torch.sum(p) + torch.sum(ground_truth_mask) - intersection
        iou = (intersection + 1e-7) / (union + 1e-7)
        iou_loss = torch.mean((iou - pred_iou) ** 2)
        return iou_loss


class FocalDiceloss_IoULoss(nn.Module):
    
    def __init__(self, weight=20.0, iou_scale=1.0):
        super(FocalDiceloss_IoULoss, self).__init__()
        self.weight = weight
        self.iou_scale = iou_scale
        self.focal_loss = FocalLoss()
        self.dice_loss = DiceLoss()
        self.maskiou_loss = MaskIoULoss()

    def forward(self, pred, mask, pred_iou):
        """
        pred: [B, 1, H, W]
        mask: [B, 1, H, W]
        """
        assert pred.shape == mask.shape, "pred and mask should have the same shape."

        focal_loss = self.focal_loss(pred, mask)
        dice_loss =self.dice_loss(pred, mask)
        loss1 = self.weight * focal_loss + dice_loss
        loss2 = self.maskiou_loss(pred, mask, pred_iou)
        loss = loss1 + loss2 * self.iou_scale
        return loss
# import os
# import numpy as np
# from sklearn.metrics import jaccard_score, f1_score
# from PIL import Image

# # 定义解剖结构名称
# anatomical_structures = [
    
#      "Right_Ventricular_Wall", "Right_Ventricle", "Right_Atrium",
#     "Right_Lung",  "Interventricular_Septum", "Left_Ventricular_Wall",
#     "Left_Ventricle", "Left_Atrium", "Left_Lung", "Interatrial_Septum",
#     "Rib",  "Spine", "Descending_Aorta"
# ]

# # 文件夹路径
# pred_folder = '/media/Storage4/yc/SAM-Med2D/workdir/Cradiac_13_147/boxes_prompt'
# mask_folder = '/media/Storage4/yc/SAM-Med2D/Cradiac_13/masks'


# # 计算IoU和Dice
# results = {}

# for filename in os.listdir(pred_folder):
#     if filename.endswith('.png'):
#         # 去掉后缀
#         base_name = filename[:-4]  # 去掉".png"
#         parts = base_name.split('_')

#         # 从后往前遍历提取解剖结构
#         # structure = None
#         # for i in range(len(parts) - 2, -1, -1):  # 从倒数第二部分开始
#         #     for j in range(i + 1, len(parts) - 1):  # 组合部分
#         #         part = '_'.join(parts[i:j + 1])
#         #         if part in anatomical_structures:
#         #             structure = part
#         #             break
#         #     if structure:
#         #         break
#         # 从后往前遍历提取解剖结构
#         # 从后往前遍历提取解剖结构
#         structure = None
#         for i in range(len(parts) - 2, -1, -1):  # 从倒数第二部分开始
#             # 检查组合的部分
#             for j in range(i + 1, len(parts) - 1):  # 从当前部分到倒数第二部分
#                 part = '_'.join(parts[i:j + 1])
#                 if part in anatomical_structures:
#                     structure = part
#                     break
#             if structure:
#                 break
#         # 检查单个部分
#         if structure is None:
#             for part in reversed(parts[:-1]):  # 排除最后的'000'
#                 if part in anatomical_structures:
#                     structure = part
#                     break




#         # 打印调试信息
#         # print(f"Processing file: {filename}, Extracted parts: {parts}, Extracted structure: {structure}")

#         if structure:
#             pred_path = os.path.join(pred_folder, filename)
#             mask_path = os.path.join(mask_folder, filename)

#             if os.path.exists(mask_path):
#                 pred = np.array(Image.open(pred_path).convert('L')) / 255
#                 mask = np.array(Image.open(mask_path).convert('L')) / 255

#                 intersection = np.sum(pred * mask)
#                 union = np.sum(pred) + np.sum(mask) - intersection
                
#                 iou = intersection / union if union != 0 else 0
#                 dice = 2 * intersection / (np.sum(pred) + np.sum(mask)) if (np.sum(pred) + np.sum(mask)) != 0 else 0
                
#                 if structure not in results:
#                     results[structure] = {'IoU': 0, 'Dice': 0, 'count': 0}
                
#                 results[structure]['IoU'] += iou
#                 results[structure]['Dice'] += dice
#                 results[structure]['count'] += 1

# # 计算总的 IoU 和 Dice
# total_iou = 0
# total_dice = 0
# total_count = 0

# # 计算平均IoU和Dice
# for structure, metrics in results.items():
#     if metrics['count'] > 0:
#         metrics['IoU'] /= metrics['count']
#         metrics['Dice'] /= metrics['count']
        
#         # 更新总的 IoU 和 Dice
#         total_iou += metrics['IoU'] * metrics['count']
#         total_dice += metrics['Dice'] * metrics['count']
#         total_count += metrics['count']
        
#         print(f"{structure}: Average IoU = {metrics['IoU']:.4f}, Average Dice = {metrics['Dice']:.4f}")

# # 计算总体平均 IoU 和 Dice
# if total_count > 0:
#     overall_iou = total_iou / total_count
#     overall_dice = total_dice / total_count
#     print(f"Overall: Average IoU = {overall_iou:.4f}, Average Dice = {overall_dice:.4f}")
# else:
#     print("No valid predictions to calculate overall metrics.")









import os
import torch
import numpy as np
from PIL import Image

# 定义解剖结构名称
# anatomical_structures = [
#     "Right_Ventricular_Wall", "Right_Ventricle", "Right_Atrium",
#     "Right_Lung", "Interventricular_Septum", "Left_Ventricular_Wall",
#     "Left_Ventricle", "Left_Atrium", "Left_Lung", "Interatrial_Septum",
#     "Rib", "Spine", "Descending_Aorta"
# ]


anatomical_structures=[
    "LV","MYO","LA"
    # "LA"
]


# 文件夹路径
pred_folder = '/media/Storage4/yc/SAM-Med2D/workdir/camus_new_132/boxes_prompt'
mask_folder = '/media/Storage4/yc/SAM-Med2D/CAMUS/mask/validation'

# 计算 IoU 和 Dice
results = {}

# 计算 IoU
def iou(pr, gt, eps=1e-7, threshold=0.5):
    pr_ = (pr > threshold).float()
    gt_ = (gt > threshold).float()
    intersection = torch.sum(gt_ * pr_, dim=[1, 2, 3])
    union = torch.sum(gt_, dim=[1, 2, 3]) + torch.sum(pr_, dim=[1, 2, 3]) - intersection
    return ((intersection + eps) / (union + eps)).cpu().numpy()

# 计算 Dice
def dice(pr, gt, eps=1e-7, threshold=0.5):
    pr_ = (pr > threshold).float()
    gt_ = (gt > threshold).float()
    intersection = torch.sum(gt_ * pr_, dim=[1, 2, 3])
    union = torch.sum(gt_, dim=[1, 2, 3]) + torch.sum(pr_, dim=[1, 2, 3])
    return ((2. * intersection + eps) / (union + eps)).cpu().numpy()

# 解析文件名中的解剖结构
# def extract_structure(filename):
#     base_name = filename[:-4]  # 去掉 ".png"
#     parts = base_name.split('_')

#     # 先尝试从倒数第二部分向前搜索
#     for i in range(len(parts) - 2, -1, -1):
#         for j in range(i + 1, len(parts) - 1):
#             part = '_'.join(parts[i:j + 1])
#             if part in anatomical_structures:
#                 return part

#     # 其次尝试单个部分匹配
#     for part in reversed(parts[:-1]):  # 排除最后的 '000'
#         if part in anatomical_structures:
#             return part

#     return None  # 没有找到解剖结构



def extract_structure(filename):
    base_name = filename[:-4]  # 去掉 ".png"
    parts = base_name.split('_')

    # 直接取最后一部分作为解剖结构名称
    structure = parts[-1]

    # 确保解析出的部分确实是一个有效的解剖结构
    if structure in anatomical_structures:
        return structure
    else:
        return None  # 未找到匹配的解剖结构



# 遍历预测结果文件
for filename in os.listdir(pred_folder):
    if filename.endswith('.png'):
        structure = extract_structure(filename)

        if structure:
            pred_path = os.path.join(pred_folder, filename)
            mask_path = os.path.join(mask_folder, filename)

            if os.path.exists(mask_path):
                # 读取预测掩码和真实掩码
                pred = torch.tensor(np.array(Image.open(pred_path).convert('L')) / 255.0, dtype=torch.float32)
                mask = torch.tensor(np.array(Image.open(mask_path).convert('L')) / 255.0, dtype=torch.float32)

                # 变换维度到 (1, 1, H, W)
                pred = pred.unsqueeze(0).unsqueeze(0)
                mask = mask.unsqueeze(0).unsqueeze(0)

                # 计算 IoU 和 Dice
                iou_score = iou(pred, mask)[0]
                dice_score = dice(pred, mask)[0]

                if structure not in results:
                    results[structure] = {'IoU': 0, 'Dice': 0, 'count': 0}

                results[structure]['IoU'] += iou_score
                results[structure]['Dice'] += dice_score
                results[structure]['count'] += 1

# 计算总的 IoU 和 Dice
total_iou = 0
total_dice = 0
total_count = 0

# 计算平均 IoU 和 Dice
for structure, metrics in results.items():
    if metrics['count'] > 0:
        metrics['IoU'] /= metrics['count']
        metrics['Dice'] /= metrics['count']
        
        total_iou += metrics['IoU'] * metrics['count']
        total_dice += metrics['Dice'] * metrics['count']
        total_count += metrics['count']
        
        print(f"{structure}: Average IoU = {metrics['IoU']:.4f}, Average Dice = {metrics['Dice']:.4f}")

# 计算总体平均 IoU 和 Dice
if total_count > 0:
    overall_iou = total_iou / total_count
    overall_dice = total_dice / total_count
    print(f"Overall: Average IoU = {overall_iou:.4f}, Average Dice = {overall_dice:.4f}")
else:
    print("No valid predictions to calculate overall metrics.")

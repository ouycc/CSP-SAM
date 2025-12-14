import os
import glob

# # 设置文件夹路径
# folder_path = "/media/Storage1/yc/mmsegmentation/Dynamic/masks"

# # 获取该文件夹下所有 .png 文件
# image_files = glob.glob(os.path.join(folder_path, "*.png"))

# # 输出图像个数
# print(f"图像个数: {len(image_files)}")


import os
import random
import shutil

# 设置文件夹路径
images_folder = "/media/Storage1/yc/mmsegmentation/Dynamic/images"
masks_folder = "/media/Storage1/yc/mmsegmentation/Dynamic/masks"
output_images_folder = "/media/Storage1/yc/mmsegmentation/Dynamic/image"
output_masks_folder = "/media/Storage1/yc/mmsegmentation/Dynamic/mask"

# 创建输出文件夹，如果不存在
os.makedirs(output_images_folder, exist_ok=True)
os.makedirs(output_masks_folder, exist_ok=True)

# 获取文件夹中的所有文件
image_files = sorted([f for f in os.listdir(images_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
mask_files = sorted([f for f in os.listdir(masks_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])

# 检查文件夹中文件是否匹配
assert len(image_files) == len(mask_files), "图像和掩码文件数目不一致"

# 随机选择 5000 个文件的索引
selected_indices = random.sample(range(len(image_files)), 5000)

# 将选中的文件复制到新的文件夹
for idx in selected_indices:
    image_file = image_files[idx]
    mask_file = mask_files[idx]

    # 复制文件到输出文件夹
    shutil.copy(os.path.join(images_folder, image_file), os.path.join(output_images_folder, image_file))
    shutil.copy(os.path.join(masks_folder, mask_file), os.path.join(output_masks_folder, mask_file))

print(f"成功选取并复制了 5000 张图像和掩码")

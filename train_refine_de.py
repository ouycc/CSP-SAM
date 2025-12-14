from segment_anything_cnn_refine_decoder_2 import sam_model_registry, SamPredictor
import torch.nn as nn
import torch
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import os
from torch import optim
from torch.utils.data import DataLoader
from DataLoader import TrainingDataset, stack_dict_batched
from utils import FocalDiceloss_IoULoss, get_logger, generate_point, setting_prompt_none,generate_uncertain_point,sam_refiner
from metrics import SegMetrics
import time
from tqdm import tqdm
import numpy as np
import datetime
from torch.nn import functional as F
# from apex import amp
import random




import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import os

def save_overlayed_masks(args, batched_input, masks, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    # 假设batched_input["image"] shape是 [B, C, H, W]
    images = batched_input["image"]  # tensor
    batch_size = images.shape[0]

    for i in range(batch_size):
        # 获取一张图和对应的mask
        image = images[i]  # [C, H, W]
        mask = masks[i]    # [1, H', W'] or [H', W']

        # 如果mask是单通道，添加一个维度
        if mask.dim() == 3:
            mask = mask.squeeze(0)

        # 将mask resize到原图大小
        original_h, original_w = image.shape[1], image.shape[2]
        mask_resized = F.interpolate(mask.unsqueeze(0).unsqueeze(0).float(), size=(original_h, original_w), mode="nearest").squeeze().cpu().numpy()

        # 将image从tensor转成PIL Image
        image_np = image.permute(1,2,0).cpu().numpy()  # [H, W, C]
        image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min() + 1e-8)  # 归一化到0-1
        image_np = (image_np * 255).astype(np.uint8)
        image_pil = Image.fromarray(image_np)

        # 构造颜色映射
        color_map = {
            1: [173, 216, 230],
            2: [128, 128, 128],
            3: [128, 128, 0],
        }

        # 创建一个彩色mask
        color_mask = np.zeros((original_h, original_w, 3), dtype=np.uint8)
        for class_id, color in color_map.items():
            color_mask[mask_resized == class_id] = color

        color_mask_pil = Image.fromarray(color_mask)

        # 将原图和mask叠加
        overlay = Image.blend(image_pil.convert("RGB"), color_mask_pil.convert("RGB"), alpha=0.5)

        # 保存
        overlay.save(os.path.join(save_dir, f"overlay_{i}.png"))








def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", type=str, default="workdir", help="work dir")
    parser.add_argument("--run_name", type=str, default="test", help="run model name")
    parser.add_argument("--epochs", type=int, default=150, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="train batch size")
    parser.add_argument("--image_size", type=int, default=256, help="image_size")
    parser.add_argument("--mask_num", type=int, default=5, help="get mask number")
    parser.add_argument("--data_path", type=str, default="/media/Storage4/yc/SAM-Med2D/CAMUS", help="train data path") 
    parser.add_argument("--metrics", nargs='+', default=['iou', 'dice'], help="metrics")
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--resume", type=str, default=None, help="load resume") 
    parser.add_argument("--model_type", type=str, default="vit_b", help="sam model_type")
    parser.add_argument("--sam_checkpoint", type=str, default="/media/Storage4/yc/SAM-Med2D/pretrain_model/sam-med2d_b.pth", help="sam checkpoint")
    parser.add_argument("--iter_point", type=int, default=8, help="point iterations")
    parser.add_argument('--lr_scheduler', type=str, default=None, help='lr scheduler')
    parser.add_argument("--point_list", type=list, default=[1, 3, 5, 9], help="point_list")
    parser.add_argument("--multimask", type=bool, default=True, help="ouput multimask")
    parser.add_argument("--encoder_adapter", type=bool, default=True, help="use adapter")
    parser.add_argument("--use_amp", type=bool, default=False, help="use amp")
    args = parser.parse_args()
    if args.resume is not None:
        args.sam_checkpoint = None
    return args


def to_device(batch_input, device):
    device_input = {}
    for key, value in batch_input.items():
        if value is not None:
            if key=='image' or key=='label':
                device_input[key] = value.float().to(device)
            elif type(value) is list or type(value) is torch.Size:
                 device_input[key] = value
            else:
                device_input[key] = value.to(device)
        else:
            device_input[key] = value
    return device_input


def prompt_and_decoder(args, batched_input, model, image_embeddings,decoder_iter = False):
    if  batched_input["point_coords"] is not None:
        points = (batched_input["point_coords"], batched_input["point_labels"])
    else:
        points = None

    if decoder_iter:
        with torch.no_grad():
            sparse_embeddings, dense_embeddings = model.prompt_encoder(
                points=points,
                boxes=batched_input.get("boxes", None),
                masks=batched_input.get("mask_inputs", None),
            )

    else:
        sparse_embeddings, dense_embeddings = model.prompt_encoder(
            points=points,
            boxes=batched_input.get("boxes", None),
            masks=batched_input.get("mask_inputs", None),
        )


    low_res_masks, iou_predictions = model.mask_decoder(
        image_embeddings = image_embeddings,
        image_pe = model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=args.multimask,
    )

    if args.multimask:
        max_values, max_indexs = torch.max(iou_predictions, dim=1)
        max_values = max_values.unsqueeze(1)
        iou_predictions = max_values
        low_res = []
        for i, idx in enumerate(max_indexs):
            low_res.append(low_res_masks[i:i+1, idx])
        low_res_masks = torch.stack(low_res, 0)

    masks = F.interpolate(low_res_masks,(args.image_size, args.image_size), mode="bilinear", align_corners=False)
    return masks, low_res_masks, iou_predictions


def train_one_epoch(args, model, optimizer, train_loader, epoch, criterion):
    train_loader = tqdm(train_loader)
    train_losses = []
    train_iter_metrics = [0] * len(args.metrics)
    for batch, batched_input in enumerate(train_loader):
        batched_input = stack_dict_batched(batched_input)
        batched_input = to_device(batched_input, args.device)
        
        # if random.random() > 0.5:
        #     batched_input["point_coords"] = None
        #     flag = "boxes"
        # else:
        #     batched_input["boxes"] = None
        #     flag = "point"

        batched_input["point_coords"]=None
        flag="boxes"

        # batched_input["boxes"] = None
        # flag = "point"

        for n, value in model.image_encoder.named_parameters():
            if "cnn_embed" in n or "post_pos_embed"  in n or "Adapter"  in n or "2.attn.rel_pos"  in n or "5.attn.rel_pos"  in n or "8.attn.rel_pos"  in n or "11.attn.rel_pos"  in n or "upneck"  in n:
                value.requires_grad = True
            else:
                value.requires_grad = False

        if args.use_amp:
            labels = batched_input["label"].half()
            image_embeddings,feature_list= model.image_encoder(batched_input["image"].half())
  
            B, _, _, _ = image_embeddings.shape
            image_embeddings_repeat = []
            for i in range(B):
                image_embed = image_embeddings[i]
                image_embed = image_embed.repeat(args.mask_num, 1, 1, 1)
                image_embeddings_repeat.append(image_embed)
            image_embeddings = torch.cat(image_embeddings_repeat, dim=0)

            masks, low_res_masks, iou_predictions = prompt_and_decoder(args, batched_input, model, image_embeddings, decoder_iter = False)
            loss = criterion(masks, labels, iou_predictions)
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward(retain_graph=False)

        else:
            labels = batched_input["label"]
            image_embeddings,cnn_feature_list= model.image_encoder(batched_input["image"])
            # image_embeddings= model.image_encoder(batched_input["image"])
            cnn_feature_list = [feat.repeat(args.mask_num, 1, 1, 1) for feat in cnn_feature_list]
            B, _, _, _ = image_embeddings.shape
            image_embeddings_repeat = []
            for i in range(B):
                image_embed = image_embeddings[i]
                image_embed = image_embed.repeat(args.mask_num, 1, 1, 1)
                image_embeddings_repeat.append(image_embed)
            image_embeddings = torch.cat(image_embeddings_repeat, dim=0)

            masks, low_res_masks, iou_predictions = prompt_and_decoder(args, batched_input, model, image_embeddings,decoder_iter = False)
            loss = criterion(masks, labels, iou_predictions)
            loss.backward(retain_graph=True)

        optimizer.step()
        optimizer.zero_grad()

        if int(batch+1) % 50 == 0:
            print(f'Epoch: {epoch+1}, Batch: {batch+1}, first {flag} prompt: {SegMetrics(masks, labels, args.metrics)}')
            # print(f'Epoch: {epoch+1}, Batch: {batch+1}, free prompt: {SegMetrics(masks, labels, args.metrics)}')

        # point_num = random.choice(args.point_list)
        # batched_input = generate_point(masks, labels, low_res_masks, batched_input, point_num)
        # # batched_input = generate_uncertain_point(masks, labels, low_res_masks, batched_input, point_num)
        # batched_input = to_device(batched_input, args.device)
    
        image_embeddings = image_embeddings.detach().clone()
        # for n, value in model.named_parameters():
        #     if "image_encoder" in n:
        #         value.requires_grad = False
        #     else:
        #         value.requires_grad = True
        coarse_masks = (torch.sigmoid(masks) > 0.5).to(torch.uint8)


        # save_overlayed_masks(args, batched_input, coarse_masks, save_dir="/media/Storage4/yc/SAM-Med2D/save/saved_masks")


        init_mask_num = np.random.randint(1, args.iter_point - 1)
        for iter in range(args.iter_point):
            refined_masks,sam_ious, sam_masks3 = sam_refiner(batched_input, coarse_masks,model,image_embeddings,cnn_feature_list)
            # refined_masks,sam_ious, sam_masks3 = sam_refiner(batched_input, coarse_masks,model,image_embeddings)
            max_values, max_indexs = torch.max(sam_ious, dim=1)
            max_values = max_values.unsqueeze(1)
            iou_predictions = max_values
            mask = []
            for i, idx in enumerate(max_indexs):
                mask.append(refined_masks[i:i+1, idx])
            refined_masks = torch.stack(mask, 0)
            loss = criterion(refined_masks, labels, sam_ious)
            coarse_masks = (torch.sigmoid(refined_masks) > 0.5).to(torch.uint8)
            loss.backward(retain_graph=True)
        optimizer.step()
        optimizer.zero_grad()
        # init_mask_num = np.random.randint(1, args.iter_point - 1)
        # for iter in range(args.iter_point):
        #     if iter == init_mask_num or iter == args.iter_point - 1:
        #         batched_input = setting_prompt_none(batched_input)

        #     if args.use_amp:
        #         masks, low_res_masks, iou_predictions = prompt_and_decoder(args, batched_input, model, image_embeddings, decoder_iter=True)
        #         loss = criterion(masks, labels, iou_predictions)
        #         with amp.scale_loss(loss,  optimizer) as scaled_loss:
        #             scaled_loss.backward(retain_graph=True)
        #     else:
        #         masks, low_res_masks, iou_predictions = prompt_and_decoder(args, batched_input, model, image_embeddings,decoder_iter=True)
        #         loss = criterion(masks, labels, iou_predictions)
        #         loss.backward(retain_graph=True)
                
        #     optimizer.step()
        #     optimizer.zero_grad()
          
        #     if iter != args.iter_point - 1:
        #         point_num = random.choice(args.point_list)
        #         batched_input = generate_point(masks, labels, low_res_masks, batched_input, point_num)
        #         # batched_input = generate_uncertain_point(masks, labels, low_res_masks, batched_input, point_num)
        #         batched_input = to_device(batched_input, args.device)
       
        #     if int(batch+1) % 50 == 0:
        #         if iter == init_mask_num or iter == args.iter_point - 1:
        #             print(f'Epoch: {epoch+1}, Batch: {batch+1}, mask prompt: {SegMetrics(masks, labels, args.metrics)}')
        #         else:
        #             print(f'Epoch: {epoch+1}, Batch: {batch+1}, point {point_num} prompt: { SegMetrics(masks, labels, args.metrics)}')

        # if int(batch+1) % 200 == 0:
        #     print(f"epoch:{epoch+1}, iteration:{batch+1}, loss:{loss.item()}")
            # save_path = os.path.join(f"{args.work_dir}/models", args.run_name, f"epoch{epoch+1}_batch{batch+1}_sam.pth")
            # state = {'model': model.state_dict(), 'optimizer': optimizer}
            # torch.save(state, save_path)

        # save_overlayed_masks(args, batched_input, coarse_masks, save_dir="/media/Storage4/yc/SAM-Med2D/save/saved_masks_2")
        train_losses.append(loss.item())

        gpu_info = {}
        gpu_info['gpu_name'] = args.device 
        train_loader.set_postfix(train_loss=loss.item(), gpu_info=gpu_info)

        train_batch_metrics = SegMetrics(masks, labels, args.metrics)
        train_iter_metrics = [train_iter_metrics[i] + train_batch_metrics[i] for i in range(len(args.metrics))]

    return train_losses, train_iter_metrics



def main(args):
    model = sam_model_registry[args.model_type](args).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = FocalDiceloss_IoULoss()

    if args.lr_scheduler:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10], gamma = 0.5)
        print('*******Use MultiStepLR')

    if args.resume is not None:
        with open(args.resume, "rb") as f:
            checkpoint = torch.load(f)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'].state_dict())
            print(f"*******load {args.resume}")

    if args.use_amp:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
        print("*******Mixed precision with Apex")
    else:
        print('*******Do not use mixed precision')

    train_dataset = TrainingDataset(args.data_path, image_size=args.image_size, mode='train', point_num=1, mask_num=args.mask_num, requires_name = False)
    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True, num_workers=4)
    print('*******Train data:', len(train_dataset))   

    loggers = get_logger(os.path.join(args.work_dir, "logs", f"{args.run_name}_{datetime.datetime.now().strftime('%Y%m%d-%H%M.log')}"))

    best_loss = 1e10
    l = len(train_loader)

    for epoch in range(0, args.epochs):
        model.train()
        train_metrics = {}
        start = time.time()
        os.makedirs(os.path.join(f"{args.work_dir}/models", args.run_name), exist_ok=True)
        # train_losses, train_iter_metrics = train_one_epoch(args, model, optimizer, train_loader, epoch, criterion)
        train_losses, train_iter_metrics = train_one_epoch(args, model, optimizer, train_loader, epoch, criterion)

        if args.lr_scheduler is not None:
            scheduler.step()

        train_iter_metrics = [metric / l for metric in train_iter_metrics]
        train_metrics = {args.metrics[i]: '{:.4f}'.format(train_iter_metrics[i]) for i in range(len(train_iter_metrics))}

        average_loss = np.mean(train_losses)
        lr = scheduler.get_last_lr()[0] if args.lr_scheduler is not None else args.lr
        loggers.info(f"epoch: {epoch + 1}, lr: {lr}, Train loss: {average_loss:.4f}, metrics: {train_metrics}")

        if average_loss < best_loss:
            best_loss = average_loss
            save_path = os.path.join(args.work_dir, "models", args.run_name, f"epoch{epoch+1}_sam.pth")
            state = {'model': model.float().state_dict(), 'optimizer': optimizer}
            torch.save(state, save_path)


        # # 仅保存最佳权重
        # if average_loss < best_loss:
        #     best_loss = average_loss
        #     save_dir = os.path.join(args.work_dir, "models", args.run_name)
        #     os.makedirs(save_dir, exist_ok=True)
        #     save_path = os.path.join(save_dir, "best_sam.pth")  # 始终覆盖保存为 best_sam.pth
        #     state = {'model': model.float().state_dict(), 'optimizer': optimizer.state_dict()}
        #     torch.save(state, save_path)
        #     loggers.info(f"Saved best model with loss {best_loss:.4f} to {save_path}")
            if args.use_amp:
                model = model.half()

        end = time.time()
        print("Run epoch time: %.2fs" % (end - start))


if __name__ == '__main__':
    args = parse_args()
    main(args)


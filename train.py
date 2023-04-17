import os
import time
import datetime
from typing import Union, List

import torch
from torch.utils import data

from src import u2net_full
from train_utils import train_one_epoch, evaluate, get_params_groups, create_lr_scheduler
from my_dataset import DUTSDataset
import transforms as T

# 对训练集进行预处理
class SODPresetTrain:
    def __init__(self, base_size: Union[int, List[int]], crop_size: int,
                 hflip_prob=0.5, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Resize(base_size, resize_mask=True),
            # 随机裁剪
            T.RandomCrop(crop_size),
            # 水平方向的随机翻转
            T.RandomHorizontalFlip(hflip_prob),
            T.Normalize(mean=mean, std=std)
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)

# 对验证集进行预处理
class SODPresetEval:
    def __init__(self, base_size: Union[int, List[int]], mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Resize(base_size, resize_mask=False),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size

    # 用来保存训练以及验证过程中信息
    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    # 读取训练集或验证集数据
    # transforms=SODPresetTrain([320, 320], crop_size=288)--预处理，将图片resize到320x320，再随机裁剪到288x288
    train_dataset = DUTSDataset(args.data_path, train=True, transforms=SODPresetTrain([320, 320], crop_size=288))
    val_dataset = DUTSDataset(args.data_path, train=False, transforms=SODPresetEval([320, 320]))

    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    train_data_loader = data.DataLoader(train_dataset,
                                        batch_size=batch_size,
                                        num_workers=num_workers,
                                        shuffle=True,
                                        pin_memory=True,
                                        collate_fn=train_dataset.collate_fn)

    val_data_loader = data.DataLoader(val_dataset,
                                      batch_size=1,  # must be 1，不然会报错
                                      num_workers=num_workers,
                                      pin_memory=True,
                                      collate_fn=val_dataset.collate_fn)
    # 创建模型
    model = u2net_full()
    # 将模型指认到对应的训练设备中
    model.to(device)

    params_group = get_params_groups(model, weight_decay=args.weight_decay)
    optimizer = torch.optim.AdamW(params_group, lr=args.lr, weight_decay=args.weight_decay)
    # 学习率变化策略，先warmup，再以cos的形式进行衰减
    lr_scheduler = create_lr_scheduler(optimizer, len(train_data_loader), args.epochs,
                                       warmup=True, warmup_epochs=2)

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

    # mae--平均绝对误差，越接近于0代表模型效果越好，默认初始设置为1.0
    # f1--max F-measure，越接近于1代表模型效果越好，默认初始设置为0.0
    current_mae, current_f1 = 1.0, 0.0
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        # 训练迭代过程，返回训练的平均损失mean_loss和当前学习率lr
        mean_loss, lr = train_one_epoch(model, optimizer, train_data_loader, device, epoch,
                                        lr_scheduler=lr_scheduler, print_freq=args.print_freq, scaler=scaler)

        save_file = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "lr_scheduler": lr_scheduler.state_dict(),
                     "epoch": epoch,
                     "args": args}
        if args.amp:
            save_file["scaler"] = scaler.state_dict()

        # eval_interval--设置为10，即每10个epochs验证一次，或者最后一个epochs也会验证
        if epoch % args.eval_interval == 0 or epoch == args.epochs - 1:
            # 每间隔eval_interval个epoch验证一次，减少验证频率节省训练时间
            mae_metric, f1_metric = evaluate(model, val_data_loader, device=device)
            mae_info, f1_info = mae_metric.compute(), f1_metric.compute()
            print(f"[epoch: {epoch}] val_MAE: {mae_info:.3f} val_maxF1: {f1_info:.3f}")
            # write into txt
            with open(results_file, "a") as f:
                # 记录每个epoch对应的train_loss、lr以及验证集各指标
                write_info = f"[epoch: {epoch}] train_loss: {mean_loss:.4f} lr: {lr:.6f} " \
                             f"MAE: {mae_info:.3f} maxF1: {f1_info:.3f} \n"
                f.write(write_info)

            # save_best
            # mae越小越好，f1越大越好，保存最好的权重
            if current_mae >= mae_info and current_f1 <= f1_info:
                torch.save(save_file, "save_weights/model_best.pth")

        # only save latest 10 epoch weights
        if os.path.exists(f"save_weights/model_{epoch-10}.pth"):
            os.remove(f"save_weights/model_{epoch-10}.pth")

        torch.save(save_file, f"save_weights/model_{epoch}.pth")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch u2net training")
    # 数据集的根目录
    parser.add_argument("--data-path", default="./", help="DUTS root")
    # 有GPU会默认使用GPU，没有GPU会使用CPU
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("-b", "--batch-size", default=16, type=int)
    # 权重衰减，设置优化器时的超参
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument("--epochs", default=360, type=int, metavar="N",
                        help="number of total epochs to train")
    # 每10个epochs验证一次
    parser.add_argument("--eval-interval", default=10, type=int, help="validation interval default 10 Epochs")
    # 学习率
    parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--print-freq', default=50, type=int, help='print frequency')
    # 是否训练中断后继续训练
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    # Mixed precision training parameters
    # 是否使用混合精度训练
    parser.add_argument("--amp", action='store_true',
                        help="Use torch.cuda.amp for mixed precision training")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists("./save_weights"):
        os.mkdir("./save_weights")

    main(args)

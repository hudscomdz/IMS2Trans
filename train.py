# coding=utf-8
import argparse
import os
import time
import logging
import random
import numpy as np
import torch
import torch.optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import ims2trans
from data.transforms import *
from data.datasets_nii import Brats_loadall_nii, Brats_loadall_test_nii
from data.data_utils import init_fn
from utils import Parser, criterions
from utils.parser import setup
from utils.lr_scheduler import LR_Scheduler, record_loss, MultiEpochsDataLoader
from predict import AverageMeter, test_softmax


parser = argparse.ArgumentParser()

parser.add_argument(
    "-batch_size", "--batch_size", default=1, type=int, help="Batch size"
)
parser.add_argument("--datapath", default=None, type=str)
parser.add_argument("--dataname", default="BRATS2018", type=str)
parser.add_argument("--savepath", default=None, type=str)
parser.add_argument("--resume", default=None, type=str)
parser.add_argument("--pretrain", default=None, type=str)
parser.add_argument("--lr", default=2e-4, type=float)
parser.add_argument("--weight_decay", default=1e-4, type=float)
parser.add_argument("--num_epochs", default=1000, type=int)
parser.add_argument("--iter_per_epoch", default=150, type=int)
parser.add_argument("--region_fusion_start_epoch", default=0, type=int)
parser.add_argument("--seed", default=1024, type=int)
parser.add_argument("--res", default=0, type=int)
path = os.path.dirname(__file__)

## parse arguments
args = parser.parse_args()
setup(args, "training")
args.train_transforms = "Compose([RandCrop3D((128,128,128)), RandomRotion(10), RandomIntensityChange((0.1,0.1)), RandomFlip(0), NumpyType((np.float32, np.int64)),])"
args.test_transforms = "Compose([NumpyType((np.float32, np.int64)),])"

ckpts = args.savepath
os.makedirs(ckpts, exist_ok=True)

###tensorboard writer
train_log_dir = "summary/train/"
writer = SummaryWriter(os.path.join(args.savepath, train_log_dir))

###modality missing mask
masks = [
    [False, False, False, True],
    [False, True, False, False],
    [False, False, True, False],
    [True, False, False, False],
    [False, True, False, True],
    [False, True, True, False],
    [True, False, True, False],
    [False, False, True, True],
    [True, False, False, True],
    [True, True, False, False],
    [True, True, True, False],
    [True, False, True, True],
    [True, True, False, True],
    [False, True, True, True],
    [True, True, True, True],
]
masks_torch = torch.from_numpy(np.array(masks))
mask_name = [
    "t2",
    "t1c",
    "t1",
    "flair",
    "t1cet2",
    "t1cet1",
    "flairt1",
    "t1t2",
    "flairt2",
    "flairt1ce",
    "flairt1cet1",
    "flairt1t2",
    "flairt1cet2",
    "t1cet1t2",
    "flairt1cet1t2",
]
print(masks_torch.int())


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    Z = size[4]

    cut_rat = np.cbrt(1.0 - lam)
    cut_w = np.int32(W * cut_rat)
    cut_h = np.int32(H * cut_rat)
    cut_z = np.int32(Z * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    cz = np.random.randint(Z)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbz1 = np.clip(cz - cut_z // 2, 0, Z)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    bbz2 = np.clip(cz + cut_z // 2, 0, Z)

    return bbx1, bby1, bbz1, bbx2, bby2, bbz2


def main():
    ##########setting seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

    ##########setting models
    if args.dataname in ["BRATS2021", "BRATS2020", "BRATS2018"]:
        num_cls = 4
    elif args.dataname == "BRATS2015":
        num_cls = 5
    else:
        print("dataset is error")
        exit(0)
    model = ims2trans.Model(num_cls=num_cls)
    print(model)
    model = torch.nn.DataParallel(model).cuda()

    ##########Setting learning schedule and optimizer
    lr_schedule = LR_Scheduler(args.lr, args.num_epochs)
    train_params = [
        {"params": model.parameters(), "lr": args.lr, "weight_decay": args.weight_decay}
    ]
    optimizer = torch.optim.Adam(
        train_params, betas=(0.9, 0.999), eps=1e-08, amsgrad=True
    )

    ##########Setting data
    if args.dataname in ["BRATS2020", "BRATS2015"]:
        train_file = "train.txt"
        test_file = "test.txt"
    elif args.dataname == "BRATS2018":
        ####BRATS2018 contains three splits (1,2,3)
        train_file = "train.txt"
        test_file = "test.txt"

    logging.info(str(args))
    train_set = Brats_loadall_nii(
        transforms=args.train_transforms,
        root=args.datapath,
        num_cls=num_cls,
        train_file=train_file,
    )
    test_set = Brats_loadall_test_nii(
        transforms=args.test_transforms, root=args.datapath, test_file=test_file
    )
    train_loader = MultiEpochsDataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        num_workers=1,
        pin_memory=True,
        shuffle=True,
        worker_init_fn=init_fn,
    )
    test_loader = MultiEpochsDataLoader(
        dataset=test_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=True
    )

    ##########Evaluate
    if args.resume is not None:
        checkpoint = torch.load(args.resume)
        logging.info("best epoch: {}".format(checkpoint["epoch"]))
        model.load_state_dict(checkpoint["state_dict"])

    ##########RESUME
    start_epoch = -1

    ##########Training
    start = time.time()
    torch.set_grad_enabled(True)
    logging.info("#############training############")
    # iter_per_epoch = args.iter_per_epoch
    iter_per_epoch = len(train_loader)
    train_iter = iter(train_loader)
    his_x = None
    his_target = None
    his_mask = None
    for epoch in range(start_epoch + 1, args.num_epochs):
        step_lr = lr_schedule(optimizer, epoch)
        writer.add_scalar("lr", step_lr, global_step=(epoch + 1))
        b = time.time()
        for i in range(iter_per_epoch):
            step = (i + 1) + epoch * iter_per_epoch
            ###Data load
            try:
                data = next(train_iter)
            except:
                train_iter = iter(train_loader)
                data = next(train_iter)
            x, target, mask = data[:3]

            x = x.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)

            model.module.is_training = True

            # generate mixed sample
            if his_x is None and his_target is None and his_mask is None:
                his_x = x
                his_target = target
                his_mask = mask

            new_x = torch.cat([his_x, x], dim=0).cuda()
            new_target = torch.cat([his_target, target], dim=0).cuda()

            lam = np.random.beta(1.0, 1.0)
            rand_index = torch.randperm(new_x.size()[0]).cuda()
            target_a = new_target
            target_b = new_target[rand_index]
            bbx1, bby1, bbz1, bbx2, bby2, bbz2 = rand_bbox(new_x.size(), lam)
            new_x[:, :, bbx1:bbx2, bby1:bby2, bbz1:bbz2] = new_x[
                rand_index, :, bbx1:bbx2, bby1:bby2, bbz1:bbz2
            ]

            # adjust lambda to exactly match pixel ratio
            lam = 1 - (
                (bbx2 - bbx1)
                * (bby2 - bby1)
                * (bbz2 - bbz1)
                / (new_x.size()[-1] * new_x.size()[-2] * new_x.size()[-3])
            )
            his_x = x
            his_target = target

            new_x1, new_x2 = torch.chunk(new_x, 2, dim=0)

            # compute output
            fuse_pred1, dis_preds1, prm_preds1 = model(new_x1, his_mask)
            fuse_pred2, dis_preds2, prm_preds2 = model(new_x2, mask)
            fuse_pred = torch.cat((fuse_pred1, fuse_pred2), dim=0)
            dis_preds = []
            for j in range(len(dis_preds1) - 1):
                dis_preds.append(torch.cat((dis_preds1[j], dis_preds2[j]), dim=0))
            dis_target = torch.cat((dis_preds1[-1], dis_preds2[-1]), dim=0)
            prm_preds = []
            for j in range(len(prm_preds1)):
                prm_preds.append(torch.cat((prm_preds1[j], prm_preds2[j]), dim=0))

            his_mask = mask

            ###Loss compute
            fuse_cross_loss = criterions.softmax_weighted_loss(
                fuse_pred, target_a, num_cls=num_cls
            ) * lam + criterions.softmax_weighted_loss(
                fuse_pred, target_b, num_cls=num_cls
            ) * (
                1.0 - lam
            )
            fuse_dice_loss = criterions.dice_loss(
                fuse_pred, target_a, num_cls=num_cls
            ) * lam + criterions.dice_loss(fuse_pred, target_b, num_cls=num_cls) * (
                1.0 - lam
            )
            fuse_loss = fuse_cross_loss + fuse_dice_loss

            dis_fdc_loss = torch.zeros(1).cuda().float()
            cont_loss = criterions.ContrastiveLoss()
            dis_lambda = 0.1  # FDC
            for dis_pred in dis_preds:
                dis_fdc_loss += cont_loss(dis_pred, dis_target)
            dis_loss = dis_lambda * dis_fdc_loss

            prm_cross_loss = torch.zeros(1).cuda().float()
            prm_dice_loss = torch.zeros(1).cuda().float()
            for prm_pred in prm_preds:
                prm_cross_loss += criterions.softmax_weighted_loss(
                    prm_pred, target_a, num_cls=num_cls
                ) * lam + criterions.softmax_weighted_loss(
                    prm_pred, target_b, num_cls=num_cls
                ) * (
                    1.0 - lam
                )
                prm_dice_loss += criterions.dice_loss(
                    prm_pred, target_a, num_cls=num_cls
                ) * lam + criterions.dice_loss(prm_pred, target_b, num_cls=num_cls) * (
                    1.0 - lam
                )
            prm_loss = prm_cross_loss + prm_dice_loss

            if epoch < args.region_fusion_start_epoch:
                loss = fuse_loss * 0.0 + dis_loss + prm_loss
            else:
                loss = fuse_loss + dis_loss + prm_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ###log
            writer.add_scalar("loss", loss.item(), global_step=step)
            writer.add_scalar(
                "fuse_cross_loss", fuse_cross_loss.item(), global_step=step
            )
            writer.add_scalar("fuse_dice_loss", fuse_dice_loss.item(), global_step=step)
            writer.add_scalar("dis_fdc_loss", dis_fdc_loss.item(), global_step=step)
            writer.add_scalar("dis_loss", dis_loss.item(), global_step=step)
            writer.add_scalar("prm_cross_loss", prm_cross_loss.item(), global_step=step)
            writer.add_scalar("prm_dice_loss", prm_dice_loss.item(), global_step=step)

            msg = "Epoch {}/{}, Iter {}/{}, Loss {:.4f}, ".format(
                (epoch + 1), args.num_epochs, (i + 1), iter_per_epoch, loss.item()
            )
            msg += "fusecross:{:.4f}, fusedice:{:.4f},".format(
                fuse_cross_loss.item(), fuse_dice_loss.item()
            )
            msg += "dis_fdc_loss:{:.4f}, dis_loss:{:.4f},".format(
                dis_fdc_loss.item(), dis_loss.item()
            )
            msg += "prmcross:{:.4f}, prmdice:{:.4f},".format(
                prm_cross_loss.item(), prm_dice_loss.item()
            )
            logging.info(msg)
        epoch_end_time = time.time() - b
        logging.info("train time per epoch: {} s".format(epoch_end_time))

        ##########model save
        file_name = os.path.join(ckpts, "model_last.pth")
        torch.save(
            {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optim_dict": optimizer.state_dict(),
            },
            file_name,
        )

        if (epoch + 1) % 50 == 0 or (epoch >= (args.num_epochs - 10)):
            file_name = os.path.join(ckpts, "model_{}.pth".format(epoch + 1))
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "optim_dict": optimizer.state_dict(),
                },
                file_name,
            )

    msg = "total time: {:.4f} hours".format((time.time() - start) / 3600)
    logging.info(msg)

    ##########Evaluate the last epoch model
    test_score = AverageMeter()
    with torch.no_grad():
        logging.info("###########test set wi/wo postprocess###########")
        for i, mask in enumerate(masks):
            logging.info("{}".format(mask_name[i]))
            dice_score = test_softmax(
                test_loader, model, dataname=args.dataname, feature_mask=mask
            )
            test_score.update(dice_score)
            with open("test_average.txt", mode="a+") as f:
                # perform file operations
                f.write("{},  {}\n".format(mask_name[i], dice_score))
        logging.info("Avg scores: {}".format(test_score.avg))
        with open("test_average.txt", mode="a+") as f:
            # perform file operations
            f.write("{}\n\n".format(test_score.avg))


if __name__ == "__main__":
    main()

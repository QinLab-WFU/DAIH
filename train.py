import json
import os
import time

import numpy as np
import torch
from loguru import logger
from timm.utils import AverageMeter

from loss import MarginLoss, MutualLoss
from _data_cm import build_loaders, get_topk, get_class_num
from _utils import (
    build_optimizer,
    calc_learnable_params,
    calc_map_eval,
    EarlyStopping,
    init,
    print_in_md,
    save_checkpoint,
    seed_everything,
    validate_smart,
    rename_output,
)
from _utils_cm import validate
from config import get_config
from network import build_model
from utils import deep_cluster, build_other_loaders, update_labels, calc_triplet_loss


def train_epoch(args, dataloaders, net, criteria, optimizer, epoch):
    tic = time.time()

    stat_meters = {}
    for x in ["cls&adv_loss", "aux&adv_loss", "img_mAP", "txt_mAP"]:
        stat_meters[x] = AverageMeter()

    dataloader_collection = [dataloaders["cls"], dataloaders["img_aux"], dataloaders["txt_aux"]]
    data_iterator = zip(*dataloader_collection)

    net.train()
    # cls(imgs, txts, labs, _), img_aux(imgs, txts, labs, _), txt_aux(imgs, txts, labs, _)
    for data in data_iterator:

        # task_probs=[1, 0.8] means: probability of run cls: 100%
        if np.random.choice(2, p=[1 - args.task_probs[0], args.task_probs[0]]):
            imgs, txts, labs, _ = data[0]  # dataloaders["cls"]
            img_feats, txt_feats = net(imgs.to(args.device), txts.to(args.device))
            labs = labs.to(args.device)

            dml_loss = calc_triplet_loss(criteria["cls"], labs, img_feats["cls"], txt_feats["cls"])

            target, source = args.adversarial.split("-")
            mut_loss1 = criteria["adv"](img_feats[target], img_feats[source])
            mut_loss2 = criteria["adv"](txt_feats[target], txt_feats[source])
            mut_loss = mut_loss1 + mut_loss2

            loss = dml_loss + args.adv_weight * mut_loss
            stat_meters["cls&adv_loss"].update(loss)

            ### Gradient Computation and Parameter Updating
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # to check overfitting
            map1 = calc_map_eval(img_feats["cls"].sign(), labs)
            stat_meters["img_mAP"].update(map1)

            map2 = calc_map_eval(txt_feats["cls"].sign(), labs)
            stat_meters["txt_mAP"].update(map2)

        # task_probs=[1, 0.8] means: probability of run aux: 80%
        if np.random.choice(2, p=[1 - args.task_probs[1], args.task_probs[1]]):
            # step-1: processing data in dataloaders["img_aux"]
            I1, _, L1, _ = data[1]  # dataloaders["img_aux"]
            L1 = L1.to(args.device)
            I1_feats = net.encode_img(I1.to(args.device))

            dml_loss1 = calc_triplet_loss(criteria["aux"], L1, I1_feats["aux"])##L1 labs

            target, source = args.adversarial.split("-")
            mut_loss1 = criteria["adv"](I1_feats[target], I1_feats[source])

            # step-2: processing data in dataloaders["txt_aux"]
            _, T2, L2, _ = data[2]  # dataloaders["txt_aux"]
            L2 = L2.to(args.device)

            T2_feats = net.encode_txt(T2.to(args.device))

            dml_loss2 = calc_triplet_loss(criteria["aux"], L2, T2_feats["aux"])

            target, source = args.adversarial.split("-")
            mut_loss2 = criteria["adv"](T2_feats[target], T2_feats[source])

            # end of steps
            dml_loss = dml_loss1 + dml_loss2
            mut_loss = mut_loss1 + mut_loss2

            loss = dml_loss + args.adv_weight * mut_loss
            stat_meters["aux&adv_loss"].update(loss)

            ### Gradient Computation and Parameter Updating
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        torch.cuda.empty_cache()

    toc = time.time()
    sm_str = ""
    for x in stat_meters.keys():
        sm_str += f"[{x}:{stat_meters[x].avg:.4f}]"
    logger.info(
        f"[Training][dataset:{args.dataset}][bits:{args.n_bits}][epoch:{epoch}/{args.n_epochs - 1}][time:{(toc - tic):.3f}]{sm_str}"
    )


def train_init(args):
    # setup net
    net, out_idx = build_model(args)

    # setup criterion
    criteria = {
        "cls": MarginLoss(args),
        "aux": MarginLoss(args, "aux"),
        "adv": MutualLoss(args),
    }

    logger.info(
        f"number of learnable params: {calc_learnable_params(net, criteria['cls'], criteria['aux'], criteria['adv'])}"
    )

    ### Move learnable parameters to GPU
    for _, loss in criteria.items():
        loss.to(args.device)

    # setup optimizer
    to_optim = [
        {"params": net.parameters(), "lr": args.lr, "weight_decay": args.wd},
        {"params": criteria["adv"].parameters(), "lr": args.adv_lr, "weight_decay": args.adv_wd},
        {"params": criteria["cls"].parameters(), "lr": args.beta_lr},
        {"params": criteria["aux"].parameters(), "lr": args.beta_lr},
    ]
    optimizer = build_optimizer(args.optimizer, to_optim)

    return net, out_idx, criteria, optimizer


def train(args, train_loader, query_loader, dbase_loader):
    net, out_idx, criteria, optimizer = train_init(args)

    train_loaders = {"cls": train_loader} | build_other_loaders(args, net)

    early_stopping = EarlyStopping()

    for epoch in range(args.n_epochs):
        train_epoch(args, train_loaders, net, criteria, optimizer, epoch)

        if (epoch + 1) % args.cluster_update_freq == 0:
            for x in ["img", "txt"]:
                shared_labels = deep_cluster(train_loaders["gen"], net, args.n_clusters, x)
                update_labels(train_loaders[f"{x}_aux"].dataset, shared_labels)

        # we monitor mAP@topk validation accuracy every 5 epochs
        if (epoch + 1) % 1 == 0 or (epoch + 1) == args.n_epochs:
            early_stop = validate_smart(
                args,
                query_loader,
                dbase_loader,
                early_stopping,
                epoch,
                model=net,
                out_idx=out_idx,
                multi_thread=args.multi_thread,
                validate_fnc=validate,
            )
            if early_stop:
                break

    if early_stopping.counter == early_stopping.patience:
        logger.info(
            f"without improvement, will save & exit, best mAP: {early_stopping.best_map:.3f}, best epoch: {early_stopping.best_epoch}"
        )
    else:
        logger.info(
            f"reach epoch limit, will save & exit, best mAP: {early_stopping.best_map:.3f}, best epoch: {early_stopping.best_epoch}"
        )

    save_checkpoint(args, early_stopping.best_checkpoint)

    return early_stopping.best_epoch, early_stopping.best_map


def main():
    init()
    args = get_config()

    if "rename" in args and args.rename:
        rename_output(args)

    dummy_logger_id = None
    rst = []
    for dataset in ["nuswide"]:
        # for dataset in ["flickr"]:
        print(f"processing dataset: {dataset}")
        args.dataset = dataset
        args.n_classes = get_class_num(dataset)
        args.topk = get_topk(dataset)

        train_loader, query_loader, dbase_loader = build_loaders(
            dataset, args.data_dir, batch_size=args.batch_size, num_workers=args.n_workers
        )

        # for hash_bit in [16, 32, 64, 128]:
        for hash_bit in [32]:
            print(f"processing hash-bit: {hash_bit}")
            seed_everything()
            args.n_bits = hash_bit

            args.save_dir = f"./output/{args.backbone}/{dataset}/{hash_bit}"
            os.makedirs(args.save_dir, exist_ok=True)
            # if any(x.endswith(".pth") for x in os.listdir(args.save_dir)):
            #     # raise Exception(f"*.pkl exists in {args.save_dir}")
            #     print(f"*.pth exists in {args.save_dir}, will pass")
            #     continue

            if dummy_logger_id is not None:
                logger.remove(dummy_logger_id)
            dummy_logger_id = logger.add(f"{args.save_dir}/train.log", mode="w", level="INFO")

            with open(f"{args.save_dir}/config.json", "w") as f:
                json.dump(vars(args), f, indent=4, sort_keys=True)

            best_epoch, best_map = train(args, train_loader, query_loader, dbase_loader)
            rst.append({"dataset": dataset, "hash_bit": hash_bit, "best_epoch": best_epoch, "best_map": best_map})

    print_in_md(rst)


if __name__ == "__main__":
    main()

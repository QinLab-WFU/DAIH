import torch
from loguru import logger
from tqdm import tqdm
from savemat import Save_mat, save_mat
from _utils import mean_average_precision


def predict(net, dataloader, out_idx=None, use_sign=True, verbose=True):
    device = next(net.parameters()).device
    net.eval()

    data_iter = tqdm(dataloader, desc="Extracting features") if verbose else dataloader
    i_embs, t_embs, labs = [], [], []

    # for imgs, txts, labs, _ in tqdm(dataloader): # CrossModal
    for batch in data_iter:
        with torch.no_grad():
            out = net(batch[0].to(device), batch[1].to(device))
        if out_idx is None:
            i_embs.append(out[0])
            t_embs.append(out[1])
        else:
            i_embs.append(out[0][out_idx[0]])
            t_embs.append(out[1][out_idx[1]])
        labs.append(batch[-2])

    i_embs = torch.cat(i_embs)
    t_embs = torch.cat(t_embs)

    if use_sign:
        i_embs = i_embs.sign()
        t_embs = t_embs.sign()

    return i_embs, t_embs, torch.cat(labs).to(device)


def validate(args, query_loader, dbase_loader, early_stopping, epoch, **kwargs):
    out_idx = kwargs.pop("out_idx", None)
    verbose = kwargs.pop("verbose", True)

    qB_i, qB_t, qL = predict(kwargs["model"], query_loader, out_idx=out_idx, verbose=verbose)
    rB_i, rB_t, rL = predict(kwargs["model"], dbase_loader, out_idx=out_idx, verbose=verbose)
    # assert (qL == qL2).all()
    map_i_t = mean_average_precision(qB_i, rB_t, qL, rL, args.topk)
    map_t_i = mean_average_precision(qB_t, rB_i, qL, rL, args.topk)
    # map_i_i = mean_average_precision(qB_i, rB_i, qL, rL, args.topk)
    # map_t_t = mean_average_precision(qB_t, rB_t, qL, rL, args.topk)
    save_mat(epoch=epoch, datasets="nuswide",query_img=qB_i, query_txt=qB_t,
             retrieval_img=rB_i, retrieval_txt=rB_t, query_labels=qL,
             retrieval_labels=rL, save_dir='.',mode_name="i2t", map=map_i_t)
    save_mat(epoch=epoch, datasets="nuswide",query_img=qB_i, query_txt=qB_t,
             retrieval_img=rB_i, retrieval_txt=rB_t, query_labels=qL,
             retrieval_labels=rL, save_dir='.',mode_name="t2i", map=map_t_i)


    map_v = (map_i_t + map_t_i) / 2
    map_k = "" if args.topk is None else f"@{args.topk}"

    del qB_i, rB_i, qB_t, rB_t, qL, rL
    torch.cuda.empty_cache()

    map_o = early_stopping.best_map
    early_stopping(epoch, map_v.item(), **kwargs)
    logger.info(
        f"[Evaluating][dataset:{args.dataset}][bits:{args.n_bits}][epoch:{epoch}/{args.n_epochs - 1}][best-mAP{map_k}:{map_o:.4f}][mAP{map_k}:{map_v:.4f}][I→T:{map_i_t:.4f}][T→I:{map_t_i:.4f}][count:{early_stopping.counter}]"
    )
    return early_stopping.early_stop

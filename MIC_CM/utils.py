import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from MIC.utils import run_kmeans, normalize_embeddings_by_same_label, update_labels
from _data_cm import init_dataset, ImageDataset, build_trans


def build_other_loaders(args, model):
    data = init_dataset(args.dataset, args.data_dir, verbose=False)

    gen_trans = build_trans("")
    aux_trans = transforms.Compose(
        [
            transforms.RandomResizedCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataloaders = {
        "gen": DataLoader(
            ImageDataset(data.train, gen_trans),
            batch_size=args.batch_size,
            num_workers=args.n_workers,
            drop_last=False,
        )
    }

    for x in ["img", "txt"]:
        dataloaders[f"{x}_aux"] = DataLoader(
            ImageDataset(data.train, aux_trans),
            batch_size=args.batch_size,
            num_workers=args.n_workers,
            shuffle=True,
            drop_last=True,
        )

        shared_labels = init_cluster(dataloaders["gen"], model, args.n_clusters, x)
        update_labels(dataloaders[f"{x}_aux"].dataset, shared_labels)

    return dataloaders


def init_cluster(dataloader, model, n_clusters, flag="img"):
    device = next(model.parameters()).device
    model.eval()

    embs, labs = [], []
    for batch in dataloader:
        with torch.no_grad():
            # out = model.encode_img(batch[0].cuda(), is_init_cluster_generation=True)
            out = getattr(model, f"encode_{flag}")(
                batch[0 if flag == "img" else 1].to(device), is_init_cluster_generation=True
            )
        embs.append(out)
        labs.append(batch[1])
    embs = torch.cat(embs)
    labs = torch.cat(labs).cuda()

    embs = normalize_embeddings_by_same_label(embs, labs)

    # cluster each feature, shape is: n_clusters x 1
    cluster_assignments = run_kmeans(embs.cpu().numpy(), n_clusters).squeeze(1).astype("float32")

    return cluster_assignments


def deep_cluster(dataloader, model, n_clusters, flag="img"):
    # computing DeepCluster-Embeddings
    device = next(model.parameters()).device
    model.eval()

    embs = []
    for batch in dataloader:
        with torch.no_grad():
            # out = model.encode_img(batch[0].cuda())["aux"]
            out = getattr(model, f"encode_{flag}")(batch[0 if flag == "img" else 1].to(device))["aux"]
        embs.extend(out.cpu().detach().numpy().tolist())
    # no normalization here
    embs = np.vstack(embs)

    cluster_assignments = run_kmeans(embs, n_clusters).squeeze(1).astype("float32")

    return cluster_assignments


def calc_triplet_loss(loss_func, labels, x1, x2=None, cross=True):
    dist_mat1 = torch.cdist(x1, x1)
    loss1, _ = loss_func(dist_mat1, labels)

    if x2 is None:
        return loss1

    dist_mat2 = torch.cdist(x2, x2)
    loss2, _ = loss_func(dist_mat2, labels)

    if not cross:
        return loss1 + loss2

    dist_mat3 = torch.cdist(x1, x2)
    loss3, _ = loss_func(dist_mat3, labels)

    dist_mat4 = torch.cdist(x2, x1)
    loss4, _ = loss_func(dist_mat4, labels)

    return loss1 + loss2 + loss3 + loss4

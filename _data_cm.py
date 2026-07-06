import configparser
import os.path as osp
import platform
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms as T

from CLIP.test import SimpleTokenizerMod


def get_class_num(dataset):
    r = {"nuswide": 21, "flickr": 24, "coco": 80}[dataset]
    return r


def get_topk(dataset):
    r = {"nuswide": None, "flickr": None, "coco": None}[dataset]
    return r


def get_concepts(name, root):
    with open(osp.join(root, name, "concepts.txt"), "r") as f:
        lines = f.read().splitlines()
    return np.array(lines)


def get_word2vecs(name, root, normalize=True):
    w2vs = np.load(osp.join(root, name, "w2v_list.npy"))
    w2vs = torch.from_numpy(w2vs)
    if normalize:
        w2vs = F.normalize(w2vs)
    return w2vs


def build_trans(usage, image_size=224):
    if usage == "train":
        step = [
            T.Resize(image_size, interpolation=Image.BICUBIC),
            T.CenterCrop(image_size),
        ]
    else:
        step = [T.Resize((image_size, image_size), interpolation=Image.BICUBIC)]
    return T.Compose(
        step
        + [
            T.ToTensor(),
            # torchvision model-zoo's image normalization is:
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # CLIP's is:
            T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
        ]
    )


def build_loaders(name, root, **kwargs):
    train_trans = build_trans("train")
    other_trans = build_trans("other")

    data = init_dataset(name, root)

    train_loader = DataLoader(ImageDataset(data.train, train_trans), shuffle=True, drop_last=True, **kwargs)
    query_loader = DataLoader(ImageDataset(data.query, other_trans), **kwargs)
    dbase_loader = DataLoader(ImageDataset(data.dbase, other_trans), **kwargs)

    return train_loader, query_loader, dbase_loader


def split_data(imgs, txts, labs, query_num=5000, train_num=10000, seed=42):
    """
    Split all data to train/query/dbase as following:
    https://github.com/QinLab-WFU/CLIP-based-Cross-Modal-Hashing
    """
    imgs = np.array(imgs)
    txts = np.array(txts)

    np.random.seed(seed=seed)

    random_idxes = np.random.permutation(len(imgs))
    query_idxes = random_idxes[:query_num]
    train_idxes = random_idxes[query_num : query_num + train_num]
    dbase_idxes = random_idxes[query_num:]  # Note: cross!

    train_data = (imgs[train_idxes].tolist(), txts[train_idxes], labs[train_idxes])
    query_data = (imgs[query_idxes].tolist(), txts[query_idxes], labs[query_idxes])
    dbase_data = (imgs[dbase_idxes].tolist(), txts[dbase_idxes], labs[dbase_idxes])

    return train_data, query_data, dbase_data


class BaseDataset(object):
    """
    Base class of dataset for CrossModal
    """

    def __init__(self, name, xxx_root, img_root, **kwargs):

        self.img_root = img_root

        self.tokenizer = SimpleTokenizerMod(max_words=kwargs.pop("max_words", 32))

        self.img_list = osp.join(xxx_root, "img_list.txt")
        self.lab_list = osp.join(xxx_root, "lab_list.npy")  # int8 for smaller space
        self.tag_list = osp.join(xxx_root, "tag_list.txt")

        self.check_before_run()

        self.imgs = self.process(self.img_list)
        self.set_img_abspath()  # 1.jpg -> /home/x/COCO/images/1.jpg

        self.txts = self.process(self.tag_list)
        self.convert_vectors()  # tags: texts -> vectors

        self.labs = self.process(self.lab_list)

        self.train, self.query, self.dbase = split_data(self.imgs, self.txts, self.labs)

        if kwargs.pop("verbose", True):
            print(f"=> {name.upper()} loaded")
            self.print_dataset_statistics()

    def check_before_run(self):
        """Check if all files are available before going deeper"""
        for x in ["img_list", "lab_list", "tag_list"]:
            p = getattr(self, x)
            if not osp.exists(p):
                raise RuntimeError("'{}' is not available".format(p))

    def get_imagedata_info(self, data):
        labs = data[2]
        n_cids = (labs.sum(axis=0) > 0).sum()
        n_imgs = len(data[0])
        return n_cids, n_imgs

    def print_dataset_statistics(self):
        n_train_cids, n_train_imgs = self.get_imagedata_info(self.train)
        n_query_cids, n_query_imgs = self.get_imagedata_info(self.query)
        n_dbase_cids, n_dbase_imgs = self.get_imagedata_info(self.dbase)

        print("Image Dataset statistics:")
        print("  -----------------------------")
        print("  subset | # images | # classes")
        print("  -----------------------------")
        print("  train  | {:8d} | {:9d}".format(n_train_imgs, n_train_cids))
        print("  query  | {:8d} | {:9d}".format(n_query_imgs, n_query_cids))
        print("  dbase  | {:8d} | {:9d}".format(n_dbase_imgs, n_dbase_cids))
        print("  -----------------------------")

    def set_img_abspath(self):
        self.imgs = [osp.join(self.img_root, x) for x in self.imgs]

    def convert_vectors(self):
        vecs = []
        for caption in self.txts:
            vec = self.tokenizer.text_to_vector(caption)
            vecs.append(vec)
        self.txts = vecs

    def process(self, xxx_path):
        if xxx_path.endswith(".npy"):
            return np.load(xxx_path).astype(np.float32)
        if xxx_path.endswith(".txt"):
            return open(xxx_path, "r").read().splitlines()
        raise NotImplementedError


class NUSWIDE(BaseDataset):

    def __init__(self, name, txt_root, img_root, verbose=True):
        super().__init__(name, txt_root, img_root, verbose=verbose)

    def set_img_abspath(self):
        path_dict = {p.stem: str(p) for p in Path(self.img_root).rglob("*.jpg")}
        for i in range(len(self.imgs)):
            key = self.imgs[i].replace(".jpg", "")
            self.imgs[i] = path_dict[key]


class COCO(NUSWIDE):

    def __init__(self, name, txt_root, img_root, verbose=True):
        super().__init__(name, txt_root, img_root, verbose=verbose)

    def set_img_abspath(self):
        path_dict = {p.stem.split("_")[-1]: str(p) for p in Path(self.img_root).rglob("*.jpg")}
        for i in range(len(self.imgs)):
            key = self.imgs[i].replace(".jpg", "")
            self.imgs[i] = path_dict[key]


_ds_factory = {"nuswide": NUSWIDE, "flickr": BaseDataset, "coco": COCO}


def init_dataset(name, root, **kwargs):

    # root = "/media/abc/bd9d3c7f-34a3-4712-85c9-cb5e93fa2427/2025061311_NDCG_MIC_cusa/_datasets"

    if name not in list(_ds_factory.keys()):
        raise KeyError('Invalid dataset, got "{}", but expected to be one of {}'.format(name, list(_ds_factory.keys())))

    xxx_root = osp.join(root, name)

    ini_loc = osp.join(root, name, "images", "location.ini")
    if osp.exists(ini_loc):
        config = configparser.ConfigParser()
        config.read(ini_loc)
        img_root = config["DEFAULT"][platform.system()]
    else:
        img_root = osp.join(root, name)

    return _ds_factory[name](name, xxx_root, img_root, **kwargs)


class ImageDataset(Dataset):
    """Image Dataset"""

    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        img, txt, lab = self.data[0][idx], self.data[1][idx], self.data[2][idx]

        # img path -> img tensor
        img = Image.open(img).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        return img, txt, lab, idx

    def get_all_labels(self):
        return torch.from_numpy(self.data[2])


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    db_name = "flickr"
    root = "./_datasets_cm"


    train_loader, query_loader, dbase_loader = build_loaders(db_name, root, batch_size=128, num_workers=4)

    for x in train_loader:
        print(x)
        break

    # dataset = init_dataset(db_name, root)
    #
    # trans = T.Compose(
    #     [
    #         # T.ToPILImage(),
    #         T.Resize([224, 224]),
    #         T.ToTensor(),
    #     ]
    # )
    #
    # train_set = ImageDataset(dataset.train, trans)
    # dataloader = DataLoader(train_set, batch_size=1, shuffle=True)
    # concepts = get_concepts(db_name, root)
    #
    # for imgs, txts, labs, _ in dataloader:
    #     print(imgs.shape)
    #     print(labs)
    #     print(txts)
    #     plt.imshow(imgs[0].numpy().transpose(1, 2, 0))
    #     titles = concepts[labs[0].nonzero().squeeze(1)]
    #     plt.title(titles)
    #     plt.show()
    #     break

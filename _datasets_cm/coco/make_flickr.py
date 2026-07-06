import os

import numpy as np

from make_nuswide import process_w2v, save_as_np, save_as_txt


def process_lab_con(n_images=25000):
    # generate multi-hot label
    label_dir = f"{root_dir}/mirflickr25k_annotations_v080"
    file_list = [x for x in os.listdir(label_dir) if "_r1" not in x and "README" not in x]
    labels = np.zeros((n_images, len(file_list)), dtype=np.int8)
    concepts = []
    for c, x in enumerate(file_list):
        concepts.append(x.split(".")[0])
        data = open(f"{label_dir}/{x}", "r").read().splitlines()
        for v in data:
            labels[int(v) - 1][c] = 1
    return labels, concepts


def process_img(n_images=25000):
    images = [f"im{i+1}.jpg" for i in range(n_images)]
    return np.array(images)


def process_tag(n_images=25000):
    tag_path = f"{root_dir}/images/meta/tags"
    tags = []
    for i in range(n_images):
        data = open(f"{tag_path}/tags{i+1}.txt", "r", encoding="utf-8").read().splitlines()
        tags.append(" ".join(data))
    return np.array(tags)


def test():
    from scipy.io import loadmat

    # x = loadmat("D:/GitHub/GCDH/data/FLICKR-25K/mir_glove.mat")["emb"]
    # print(x.shape)
    # print(x.dtype)
    # # plant_life -> plant
    # search_glove_by_attr(x[14])

    y = loadmat("D:/GitHub/CLIP-based-Cross-Modal-Hashing/dataset/flickr/label.mat")["category"]
    t = loadmat("D:/GitHub/CLIP-based-Cross-Modal-Hashing/dataset/flickr/caption.mat")["caption"]
    x = loadmat("D:/GitHub/CLIP-based-Cross-Modal-Hashing/dataset/flickr/index.mat")["index"]
    # print(x.shape)
    # print(x.dtype)
    for i in range(10):
        print(x[i])
        print(y[i])
        print(t[i])
        print("-" * 10)


if __name__ == "__main__":
    root_dir = "D:/Test/Datasets/MIRFLICKR-25K"
    curr_dir = os.path.dirname(__file__)

    lab_list, con_list = process_lab_con()
    img_list = process_img()
    tag_list = process_tag()

    w2v_list = process_w2v(con_list, {"plant": "plant_life"})
    # print(w2vs.shape)
    # print(concepts.index("plant_life"))

    # remove all-zero labels
    idxes = (lab_list.sum(axis=1) > 0).nonzero()[0]
    lab_list = lab_list[idxes]
    img_list = img_list[idxes]
    tag_list = tag_list[idxes]

    save_as_txt(img_list, "img_list")
    save_as_txt(con_list, "concepts")
    save_as_txt(tag_list, "tag_list")
    save_as_np(w2v_list, "w2v_list")
    save_as_np(lab_list, "lab_list")

    # test()

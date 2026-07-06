import os

import numpy as np
import pandas as pd
import scipy.io as scio

con_list = [
    "animal",
    "beach",
    "buildings",
    "clouds",
    "flowers",
    "grass",
    "lake",
    "mountain",
    "ocean",
    "person",
    "plants",
    "reflection",
    "road",
    "rocks",
    "sky",
    "snow",
    "sunset",
    "tree",
    "vehicle",
    "water",
    "window",
]


def save_as_txt(obj, obj_name):
    curr_dir = os.path.dirname(__file__)
    with open(f"{curr_dir}/{obj_name}.txt", "w", encoding="utf-8") as f:
        for x in obj:
            if isinstance(x, list):
                x = " ".join(x)
            f.write(f"{x}\n")


def save_as_np(obj, obj_name):
    curr_dir = os.path.dirname(__file__)
    np.save(f"{curr_dir}/{obj_name}.npy", obj)


def save_as_mat(obj, obj_name):
    curr_dir = os.path.dirname(__file__)
    scio.savemat(f"{curr_dir}/{obj_name}.mat", {obj_name: obj})


def save_as_csv(obj, obj_name):
    curr_dir = os.path.dirname(__file__)
    df = pd.DataFrame(obj, columns=["obj_name"])
    df.to_csv(f"{curr_dir}/{obj_name}.csv", index=False, header=False)


def process_img():
    # image_path = f"{root_dir}/Flickr"

    with open(f"{root_dir}/ImageList/Imagelist.txt", "r") as f:
        image_list = f.readlines()

    image_list = [item.strip().split("\\")[1] for item in image_list]
    print("indexs length:", len(image_list))

    return np.array(image_list)


def process_lab(n_images=269648):
    # use top 21 classes
    # with open(f"{curr_dir}/used_label.txt", encoding="utf-8") as f:
    #     picked_classes = f.readlines()
    # picked_classes = [item.strip() for item in picked_classes]

    # generate multi-hot label
    labels = np.zeros((n_images, len(con_list)), dtype=np.int8)

    label_dir = f"{root_dir}/Groundtruth/AllLabels"
    for c, x in enumerate(con_list):
        with open(f"{label_dir}/Labels_{x}.txt", "r") as f:
            data = f.readlines()
        for i, v in enumerate(data):
            labels[i][c] = 1 if v.strip() == "1" else 0
    return labels


def process_tag():
    tags = []
    with open(f"{root_dir}/NUS_WID_Tags/All_Tags.txt", "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if len(line.strip()) == 0:
                raise Exception("some line empty!")
            tag = line.split()[1:]
            if len(tag) != 0:
                tag = " ".join(tag).strip()
            else:
                print(i, line)
            tags.append(tag)
    return np.array(tags)


def process_w2v(concepts, word_dict=None):
    """
    word_dict: {key_in_glove: key_in_concept}
    """

    w2v_dict = {x: None for x in concepts}
    # print(w2v_dict.keys())

    counter = 0
    with open("D:/Test/datasets/glove.6B.300d.txt", "r", encoding="utf8") as fp:
        for line in fp:
            tmp = line.strip().split()
            if tmp[0] in w2v_dict.keys():
                w2v_dict[tmp[0]] = tmp[1:]
                counter += 1
            elif word_dict and tmp[0] in word_dict.keys():
                w2v_dict[word_dict[tmp[0]]] = tmp[1:]
                counter += 1
            else:
                continue
            if counter == len(concepts):
                break

    if counter != len(concepts):
        for k, v in w2v_dict.items():
            if v is None:
                print(f"[{k}] not found!")
        raise Exception(f"{len(concepts)-counter} classes unmatched!")

    # float32 will be accurate enough
    # 找到最长字符串的索引
    # max_index = np.unravel_index(np.argmax(np.vectorize(len)(w2v_list)), w2v_list.shape)
    # print(max_index) # (4, 259)
    return np.array([w2v_dict[x] for x in concepts], dtype=np.float32)


def search_glove_by_attr(attr, glove_type="wiki"):
    # https://nlp.stanford.edu/projects/glove/
    if glove_type == "twitter":
        file_path = "D:/Test/Datasets/glove.twitter.27B.200d.txt"
    elif glove_type == "wiki":
        file_path = "D:/Test/Datasets/glove.6B.300d.txt"
    else:
        raise Exception(f"not support: {glove_type}")
    with open(file_path, "r", encoding="utf8") as fp:
        for line in fp:
            tmp = line.rstrip().split()
            x = np.array(tmp[1:], dtype=attr.dtype)
            if (x == attr).all():
                return tmp[0]
    return None


def test():
    # with open(f"{curr_dir}/tag_list.txt", encoding="utf-8") as f:
    #     tag_list = f.readlines()

    # print(len(tag_list))

    # for i, x in enumerate(tag_list):
    #     if "123456" == x.strip():
    #         print(i)

    # lab_list = np.load(f"{curr_dir}/lab_list.npy")
    # print(type(lab_list))
    # print(lab_list[122070].nonzero()[0])

    # print((lab_list.sum(axis=1) > 1).nonzero()[0])
    import h5py

    tags = np.array(h5py.File("D:/GitHub/GCDH/data/NUS-WIDE-TC10/tagList.mat")["YAll"]).T
    print(tags.shape)
    print(tags[10])


if __name__ == "__main__":
    curr_dir = os.path.dirname(__file__)
    root_dir = "D:/Test/Datasets/NUS-WIDE"

    img_list = process_img()
    lab_list = process_lab()
    tag_list = process_tag()
    w2v_list = process_w2v(con_list)
    # print(captions[168855])  # mark as 123456

    # remove all-zero labels
    idxes = (lab_list.sum(axis=1) > 0).nonzero()[0]
    lab_list = lab_list[idxes]
    img_list = img_list[idxes]
    tag_list = tag_list[idxes]

    save_as_txt(img_list, "img_list")  # np: 14.1M -> txt: 3.8M
    save_as_txt(con_list, "concepts")
    save_as_txt(tag_list, "tag_list")  # np: 4.3G
    save_as_np(w2v_list, "w2v_list")
    save_as_np(lab_list, "lab_list")

    test()

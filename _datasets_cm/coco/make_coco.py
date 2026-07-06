import json
import os

import numpy as np

from make_nuswide import process_w2v, save_as_np, save_as_txt, search_glove_by_attr


def make_dicts(json_data, index_dict):
    result = []
    for name in index_dict:
        data = json_data[name]
        middle_dict = {}
        for item in data:
            k = item[index_dict[name][0]]
            v = item[index_dict[name][1]]
            if isinstance(v, str):
                v = v.strip().replace("\n", "")
            if k not in middle_dict:
                middle_dict.update({k: [v]})
            else:
                middle_dict[k].append(v)
        result.append(middle_dict)
    return result


def remove_unused(_dict, keeps):
    # remove ids without categories
    keys = list(_dict.keys())
    for k in keys:
        if k not in keeps:
            _dict.pop(k)


def dict_to_list(_dict):
    return [_dict[k] for k in sorted(list(_dict.keys()))]


def process_img_tag():
    img_dict, tag_dict = {}, {}
    for json_file in ["captions_train2017.json", "captions_val2017.json"]:
        with open(f"{root_dir}/annotations/{json_file}", "r") as f:
            json_data = json.load(f)

        index_dict = {"images": ["id", "file_name"], "annotations": ["image_id", "caption"]}

        result = make_dicts(json_data, index_dict)

        img_dict |= result[0]
        tag_dict |= result[1]

    # check images exist
    for k, v in img_dict.items():
        if not os.path.exists(f"{root_dir}/images/{v[0]}"):
            print(k, v)
            img_dict.pop(k)

    assert len(img_dict) == len(tag_dict)

    return img_dict, tag_dict


def process_lab_img_con():
    concepts = []
    category_dict = {}  # {category_id: class_index}
    lab_dict, img_dict = {}, {}
    for json_file in ["instances_train2017.json", "instances_val2017.json"]:
        with open(f"{root_dir}/annotations/{json_file}", "r") as f:
            json_data = json.load(f)

        if len(concepts) == 0:
            # coco中的categories（1到90）不连续，需要重新分配class（80个）
            for i, k in enumerate(json_data["categories"]):
                category_dict.update({k["id"]: i})
                concepts.append(k["name"])
        else:
            for k in json_data["categories"]:
                if k["id"] not in category_dict.keys():
                    raise Exception(f"need {k['id']}-{k['name']}")

        index_dict = {"annotations": ["image_id", "category_id"], "images": ["id", "file_name"]}

        # lab_dict = {image_id: [category_id1, c_id2, ...], ...}
        result = make_dicts(json_data, index_dict)
        lab_dict |= result[0]
        img_dict |= result[1]

        print(len(lab_dict), len(img_dict))

    # mark multi-hot label for images
    # change lab_dict
    # from: {image_id: [category_id1, c_id2, ...], ...}
    # to:   {image_id: multi-hot label}
    for k in lab_dict:
        # lab = [0] * len(category_dict)
        lab = np.zeros((len(category_dict),), dtype=np.int8)
        for category_id in lab_dict[k]:
            lab[category_dict[category_id]] = 1
        lab_dict[k] = lab

    return lab_dict, img_dict, concepts


def test():
    to_match = [
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "sports ball",
        "baseball bat",
        "baseball glove",
        "tennis racket",
        "wine glass",
        "hot dog",
        "potted plant",
        "dining table",
        "cell phone",
        "teddy bear",
        "hair drier",
    ]

    from scipy.io import loadmat

    w2vs = loadmat("D:/GitHub/GCDH/data/coco2014/coco_glove.mat")["emb"]
    print(w2vs.shape)

    con_list = open(f"{os.path.dirname(__file__)}/con_list.txt", "r", encoding="utf-8").read().splitlines()
    con_list.sort()

    for x in to_match:
        i = con_list.index(x)
        rst = search_glove_by_attr(w2vs[i])
        if rst is None:
            print(f"'{x}'")
            print(w2vs[i])
        else:
            print(f"'{x}':'{rst}'")

    # for x in w2vs:
    #     # i = con_list.index(k)
    #     # print(i)
    #     # print(f"'{k}':'{search_glove_by_attr(x)}'")
    #     print(f"'{search_glove_by_attr(x)}'")


def test2():
    _dict = {
        "baseball bat": "bat",
    }

    from scipy.io import loadmat

    w2vs = loadmat("D:/GitHub/GCDH/data/coco2014/coco_glove.mat")["emb"]
    print(w2vs.shape)

    con_list = open(f"{os.path.dirname(__file__)}/con_list.txt", "r", encoding="utf-8").read().splitlines()
    con_list.sort()

    def search(kw):
        with open("D:/Test/datasets/glove.6B.300d.txt", "r", encoding="utf8") as fp:
            for line in fp:
                tmp = line.strip().split()
                if tmp[0] == kw:
                    return tmp[1:]

    for k, v in _dict.items():
        print(k, v)
        i = con_list.index(k)
        print(search(v))
        print("-" * 10)
        print(w2vs[i])
        break


if __name__ == "__main__":
    root_dir = "D:/Test/Datasets/MS-COCO"

    img_dict, tag_dict = process_img_tag()

    lab_dict, _, con_list = process_lab_img_con()

    remove_unused(tag_dict, lab_dict.keys())
    remove_unused(img_dict, lab_dict.keys())

    assert len(lab_dict) == len(tag_dict)
    assert len(lab_dict) == len(img_dict)

    img_list = dict_to_list(img_dict)
    tag_list = dict_to_list(tag_dict)
    lab_list = dict_to_list(lab_dict)
    word_dict = {
        "table": "dining table",
        "houseplant": "potted plant",
        "ball": "sports ball",
        "light": "traffic light",
        "racket": "tennis racket",
        "glove": "baseball glove",
        "bat": "baseball bat",
        "sign": "stop sign",
        "meter": "parking meter",
        "dryer": "hair drier",
        "hydrant": "fire hydrant",
        "wineglass": "wine glass",
        "hotdog": "hot dog",
        "phone": "cell phone",
        "teddy": "teddy bear",
    }
    w2v_list = process_w2v(con_list, word_dict)

    save_as_txt(img_list, "img_list")
    save_as_txt(con_list, "concepts")
    # save_as_mat(tag_list, "tag_list")
    # save_as_csv(tag_list, "tag_list")
    save_as_txt(tag_list, "tag_list")
    save_as_np(w2v_list, "w2v_list")
    save_as_np(lab_list, "lab_list")

import argparse
from os import path as osp


def get_config():
    parser = argparse.ArgumentParser(description=osp.basename(osp.dirname(__file__)))

    # common settings
    parser.add_argument("--backbone", type=str, default="clip", help="see network.py")
    parser.add_argument("--data-dir", type=str, default="../_datasets_cm", help="directory to dataset")
    parser.add_argument("--n-epochs", type=int, default=100, help="number of epochs to train for")
    parser.add_argument("--n-workers", type=int, default=4, help="number of dataloader workers")
    parser.add_argument("--batch-size", type=int, default=128, help="batch size for training")
    parser.add_argument("--optimizer", type=str, default="adam", help="sgd/rmsprop/adam/amsgrad/adamw")
    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")
    parser.add_argument("--wd", type=float, default=4e-4, help="weight decay")
    parser.add_argument("--device", type=str, default="cuda:0", help="device (accelerator) to use")
    parser.add_argument("--multi-thread", type=bool, default=True, help="use a separate thread for validation")

    # changed at runtime
    parser.add_argument("--dataset", type=str, default="flickr", help="nuswide/flickr/coco")
    parser.add_argument("--n-classes", type=int, default=24, help="number of dataset classes")
    parser.add_argument("--topk", type=int, default=None, help="mAP@topk")
    parser.add_argument("--save-dir", type=str, default="./output", help="directory to output results")
    parser.add_argument("--n-bits", type=int, default=32, help="length of hashing binary")

    # special settings
    parser.add_argument("--tasks", default=["cls", "aux"], nargs="+", type=str, help="Name of [main task, aux. task]")
    parser.add_argument(
        "--task_probs",
        nargs="+",
        type=float,
        default=[1, 0.8],
        help="Prob. of [main task, aux. task] to be included in one iteration.",
    )
    parser.add_argument(
        "--aux_dim",
        default=32,
        type=int,
        help="Output embedding sizes of the respective embeddings. List of values for [main task<-n_bits, aux. task].",
    )

    ### Adversarial Loss function parameters (Projection Network R)
    parser.add_argument(
        "--adversarial",
        default="cls-aux",
        type=str,
        help="Directions of adversarial loss ['target-source']: 'cls-aux' (as used in the paper) and 'aux-cls'. Can contain both directions.",
    )
    parser.add_argument("--adv_lr", default=1e-5, type=float, help="learning rate for adversarial loss")
    parser.add_argument("--adv_wd", default=1e-6, type=float, help="weight decay for adversarial loss")
    parser.add_argument(
        "--adv_weight",
        default=10,
        type=float,
        help="Weighting parameter for adversarial loss. Needs to be the same length as the number of adv. loss directions.",
    )
    parser.add_argument(
        "--adv_dim", default=512, type=int, help="Dimension of linear layers in adversarial projection network."
    )

    ### Interclass Mining: Parameters
    parser.add_argument(
        "--n_clusters", default=20, type=int, help="Number of clusters for auxiliary inter-class mining task."
    )
    parser.add_argument(
        "--cluster_update_freq",
        default=3,
        type=int,
        help="Number of epochs to train before updating cluster labels. E.g. 1 -> every other epoch.",
    )

    ### DistanceWeightedMiner
    parser.add_argument(
        "--miner_distance_lower_cutoff",
        default=0.5,
        type=float,
        help="Lower cutoff on distances - values below are sampled with equal prob.",
    )
    parser.add_argument(
        "--miner_distance_upper_cutoff",
        default=1.4,
        type=float,
        help="Upper cutoff on distances - values above are IGNORED.",
    )

    ### MarginLoss
    parser.add_argument(
        "--beta_lr",
        default=5e-4,
        type=float,
        help="MARGIN: Learning rate for beta-margin values.",
    )
    parser.add_argument("--beta", default=1.2, type=float, help="MARGIN: Initial beta-margin values.")
    parser.add_argument(
        "--margin", type=float, default=0.2, help="TRIPLETS: Fixed Margin value for Triplet-based loss functions."
    )

    args = parser.parse_args()

    # mod
    # args.rename = True
    # args.task_probs = [1.0, 0.0]
    # args.cluster_update_freq = 1000
    # args.lr_mutual = 0.0
    # args.adv_weight = 0.0

    return args

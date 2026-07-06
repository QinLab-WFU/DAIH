import os
from argparse import Namespace

import torch
import torch.nn.functional as F
from torch import nn

from CLIP import clip


def build_model(args: Namespace):
    if args.backbone != "clip":
        raise NotImplementedError(f"not support: {args.backbone}")
    net = ClipMod(args.tasks, [args.n_bits, args.n_bits])
    return net.to(args.device), ["cls", "cls"]


class ClipMod(nn.Module):
    def __init__(self, out_modes, embed_dims):
        super().__init__()

        self.out_modes = out_modes

        embed_dim, self.clip = self.load_clip()

        ### Set Embedding Layer
        self.img_fc = nn.ModuleDict({task: nn.Linear(embed_dim, embed_dims[i]) for i, task in enumerate(out_modes)})
        self.txt_fc = nn.ModuleDict({task: nn.Linear(embed_dim, embed_dims[i]) for i, task in enumerate(out_modes)})

    def load_clip(self) -> tuple:
        clip_path = os.path.expanduser("~/.cache/clip/ViT-B-32.pt")
        if not os.path.exists(clip_path):
            print("downloading clip by running DCHMT's network.py")
            raise FileNotFoundError(clip_path)
        model = torch.jit.load(clip_path, map_location="cpu")
        state_dict = model.state_dict()
        # change context_length: 77 -> 32
        state_dict["positional_embedding"] = state_dict["positional_embedding"][:32,]
        embed_dim, net = state_dict["text_projection"].shape[1], clip.build_model(state_dict)
        # Half -> Float
        net.float()
        return embed_dim, net

    def encode_img(self, x, is_init_cluster_generation=False):
        x = self.clip.encode_image(x)

        if is_init_cluster_generation:
            # If the first clusters before standardization are computed: We use the initial layers with strong
            # average pooling. Using these, we saw much better initial grouping then when using layer combinations or
            # only the last layer.
            x = F.normalize(x)
            return x

        out_dict = {}
        for out_mode in self.out_modes:
            mod_x = self.img_fc[out_mode](x)
            out_dict[out_mode] = F.normalize(mod_x, dim=-1)
        return out_dict

    def encode_txt(self, x, is_init_cluster_generation=False):
        x = self.clip.encode_text(x)

        if is_init_cluster_generation:
            # If the first clusters before standardization are computed: We use the initial layers with strong
            # average pooling. Using these, we saw much better initial grouping then when using layer combinations or
            # only the last layer.
            x = F.normalize(x)
            return x

        out_dict = {}
        for out_mode in self.out_modes:
            mod_x = self.txt_fc[out_mode](x)
            out_dict[out_mode] = F.normalize(mod_x, dim=-1)
        return out_dict

    def forward(self, imgs, txts):
        return self.encode_img(imgs), self.encode_txt(txts)


if __name__ == "__main__":
    net, _ = build_model(
        Namespace(
            backbone="clip",
            tasks=["cls", "aux"],
            n_bits=16,
            aux_dim=16,
            device="cpu",
        )
    )
    x1 = torch.randn(2, 3, 224, 224)
    x2 = torch.randint(100, (2, 32))
    z1, z2 = net(x1, x2)
    print(z1)
    print(z2)

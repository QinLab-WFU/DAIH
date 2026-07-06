import math
from collections import Counter
from typing import Any

import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis
from fvcore.nn.jit_handles import get_shape

from config import get_config
from network import build_model
from _data_cm import get_class_num, get_topk
from MIC.loss import MutualLoss


# ============================================================
# 1. Basic settings
# ============================================================

DATASET = "nuswide"
HASH_BITS = 32
INPUT_SHAPE = (1, 3, 224, 224)


# ============================================================
# 2. Image-side wrappers
# ============================================================

class CLIPImagePath(nn.Module):
    """
    Profile the base CLIP-B/32 image-side representation path.

    In the existing DAIH implementation, setting
    is_init_cluster_generation=True returns the image representation
    before the task-specific class and auxiliary mapping layers.
    """

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, images: torch.Tensor):
        return self.net.encode_img(
            images,
            is_init_cluster_generation=True,
        )


class DAIHImagePath(nn.Module):
    """
    Profile the complete DAIH image-side encoding path.

    The default encode_img() call computes the image-side class branch
    and auxiliary branch. The text modality is intentionally excluded.
    """

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, images: torch.Tensor):
        outputs = self.net.encode_img(images)

        if not isinstance(outputs, dict):
            return outputs

        if "cls" not in outputs:
            raise KeyError(
                "encode_img() does not return the required key 'cls'. "
                f"Available keys: {list(outputs.keys())}"
            )

        if "aux" in outputs:
            return outputs["cls"], outputs["aux"]

        return outputs["cls"]


# ============================================================
# 3. FLOPs counting for scaled dot-product attention
# ============================================================

def sdpa_flop_jit(inputs, outputs):
    """
    Count the matrix-multiplication FLOPs in:
        aten::scaled_dot_product_attention

    fvcore follows the convention:
        one fused multiply-add = one FLOP.

    The counted operations are:
        Q @ K^T
        Attention @ V
    """
    q_shape = get_shape(inputs[0])
    k_shape = get_shape(inputs[1])
    v_shape = get_shape(inputs[2])

    if q_shape is None or k_shape is None or v_shape is None:
        return Counter()

    if len(q_shape) < 3 or len(k_shape) < 3 or len(v_shape) < 3:
        return Counter()

    # Typical shape:
    # Q: [batch, heads, query_tokens, head_dim]
    # K: [batch, heads, key_tokens, head_dim]
    # V: [batch, heads, key_tokens, value_dim]
    batch_heads = math.prod(q_shape[:-2])

    query_tokens = q_shape[-2]
    key_tokens = k_shape[-2]

    query_dim = q_shape[-1]
    value_dim = v_shape[-1]

    qk_flops = (
            batch_heads
            * query_tokens
            * key_tokens
            * query_dim
    )

    av_flops = (
            batch_heads
            * query_tokens
            * key_tokens
            * value_dim
    )

    return Counter(
        {
            "scaled_dot_product_attention": qk_flops + av_flops
        }
    )


# ============================================================
# 4. Utility functions
# ============================================================

def flatten_tensor_outputs(outputs: Any) -> list[torch.Tensor]:
    """
    Recursively collect tensors from nested model outputs.
    """
    if torch.is_tensor(outputs):
        return [outputs]

    if isinstance(outputs, dict):
        tensors = []

        for value in outputs.values():
            tensors.extend(flatten_tensor_outputs(value))

        return tensors

    if isinstance(outputs, (tuple, list)):
        tensors = []

        for value in outputs:
            tensors.extend(flatten_tensor_outputs(value))

        return tensors

    return []


def count_active_path_parameters(
        model: nn.Module,
        dummy_input: torch.Tensor,
) -> tuple[int, int]:
    """
    Count parameters that actually participate in the selected image-side
    forward path.

    The method combines:
        1. executed-module tracing;
        2. backward-gradient tracing.

    This avoids incorrectly counting the text-side parameters stored in
    the complete multimodal network.
    """
    executed_modules = set()
    handles = []

    def record_execution(module, _inputs, _outputs):
        executed_modules.add(module)

    for module in model.modules():
        handles.append(
            module.register_forward_hook(record_execution)
        )

    model.zero_grad(set_to_none=True)

    outputs = model(dummy_input)

    tensor_outputs = flatten_tensor_outputs(outputs)

    if len(tensor_outputs) == 0:
        raise RuntimeError(
            "No tensor output was returned by the selected image-side path."
        )

    scalar = sum(
        output.float().sum()
        for output in tensor_outputs
    )

    scalar.backward()

    for handle in handles:
        handle.remove()

    active_parameters = {}

    # Collect direct parameters from executed modules.
    for module in executed_modules:
        for parameter in module.parameters(recurse=False):
            active_parameters[id(parameter)] = parameter

    # Add parameters that received gradients, including parameters used
    # functionally rather than through a standard submodule forward call.
    for parameter in model.parameters():
        if parameter.grad is not None:
            active_parameters[id(parameter)] = parameter

    total_parameters = sum(
        parameter.numel()
        for parameter in active_parameters.values()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in active_parameters.values()
        if parameter.requires_grad
    )

    model.zero_grad(set_to_none=True)

    return total_parameters, trainable_parameters


def profile_image_path(
        name: str,
        model: nn.Module,
        dummy_image: torch.Tensor,
) -> dict:
    """
    Profile one image-side forward path.
    """
    model.eval()

    total_parameters, trainable_parameters = (
        count_active_path_parameters(
            model=model,
            dummy_input=dummy_image,
        )
    )

    flop_analysis = FlopCountAnalysis(
        model,
        (dummy_image,),
    )

    # Add FLOPs for fused attention kernels that fvcore does not count
    # automatically.
    flop_analysis.set_op_handle(
        "aten::scaled_dot_product_attention",
        sdpa_flop_jit,
    )

    # Ignore element-wise and shape-only operations to follow the
    # conventional matrix-multiplication and convolution FLOPs protocol.
    ignored_operators = [
        "aten::add",
        "aten::div",
        "aten::unflatten",
        "aten::mul",
        "aten::sigmoid",
        "aten::linalg_vector_norm",
        "aten::clamp_min",
        "aten::expand_as",
    ]

    for operator in ignored_operators:
        flop_analysis.set_op_handle(operator, None)

    flop_analysis.unsupported_ops_warnings(False)
    flop_analysis.uncalled_modules_warnings(False)

    total_flops = flop_analysis.total()
    unsupported_operators = dict(
        flop_analysis.unsupported_ops()
    )

    print("=" * 76)
    print(name)
    print("=" * 76)
    print(f"Input shape:               {tuple(dummy_image.shape)}")
    print(f"Image-side FLOPs:          {total_flops / 1e6:.4f} M")
    print(f"Image-side parameters:     {total_parameters / 1e6:.4f} M")
    print(f"Trainable image params:    {trainable_parameters / 1e6:.4f} M")
    print(f"Unsupported operators:     {unsupported_operators}")

    return {
        "name": name,
        "flops": total_flops,
        "parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "unsupported_operators": unsupported_operators,
    }


def print_difference(
        baseline_result: dict,
        daih_result: dict,
) -> None:
    """
    Print the additional image-side overhead introduced by DAIH.
    """
    delta_flops = (
            daih_result["flops"]
            - baseline_result["flops"]
    )

    delta_parameters = (
            daih_result["parameters"]
            - baseline_result["parameters"]
    )

    print("=" * 76)
    print("ADDITIONAL IMAGE-SIDE OVERHEAD INTRODUCED BY DAIH")
    print("=" * 76)
    print(f"Additional FLOPs:          {delta_flops / 1e6:.4f} M")
    print(f"Additional parameters:     {delta_parameters / 1e6:.4f} M")


# ============================================================
# 5. Main
# ============================================================

def main():
    args = get_config()

    args.dataset = DATASET
    args.n_classes = get_class_num(args.dataset)
    args.topk = get_topk(args.dataset)
    args.n_bits = HASH_BITS

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    args.device = device

    torch.manual_seed(0)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    # FLOPs and parameter counts depend on the network architecture,
    # not on trained checkpoint weights.
    net, _ = build_model(args)

    net = net.to(device)
    net.eval()

    dummy_image = torch.randn(
        *INPUT_SHAPE,
        device=device,
    )

    clip_image_path = CLIPImagePath(net).to(device).eval()
    daih_image_path = DAIHImagePath(net).to(device).eval()

    clip_result = profile_image_path(
        name="BASE CLIP-B/32 IMAGE-SIDE PATH",
        model=clip_image_path,
        dummy_image=dummy_image,
    )

    daih_result = profile_image_path(
        name="COMPLETE DAIH IMAGE-SIDE PATH",
        model=daih_image_path,
        dummy_image=dummy_image,
    )

    print_difference(
        baseline_result=clip_result,
        daih_result=daih_result,
    )

    # The projection network R is used only by the adversarial mutual
    # information loss during training. It is reported separately because
    # it does not participate in online I2T image encoding.
    mutual_loss = MutualLoss(args).to(device)

    projection_parameters = sum(
        parameter.numel()
        for parameter in mutual_loss.parameters()
    )

    print("-" * 76)
    print("TRAINING-ONLY ADVERSARIAL MODULE")
    print("-" * 76)
    print(
        f"Projection-network parameters: "
        f"{projection_parameters / 1e6:.4f} M"
    )
    print("=" * 76)

    print("\nSUMMARY FOR PAPER TABLE")
    print("-" * 76)
    print(
        f"CLIP-B/32 image path: "
        f"{clip_result['flops'] / 1e6:.2f} M FLOPs, "
        f"{clip_result['parameters'] / 1e6:.2f} M parameters"
    )
    print(
        f"DAIH image path:      "
        f"{daih_result['flops'] / 1e6:.2f} M FLOPs, "
        f"{daih_result['parameters'] / 1e6:.2f} M parameters"
    )
    print(
        f"DAIH increment:       "
        f"{(daih_result['flops'] - clip_result['flops']) / 1e6:.2f} M FLOPs, "
        f"{(daih_result['parameters'] - clip_result['parameters']) / 1e6:.2f} M parameters"
    )
    print("-" * 76)


if __name__ == "__main__":
    main()

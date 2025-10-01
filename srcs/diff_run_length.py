import torch
import os
import gc

from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from srcs.quantizer.real_quantize import *
from srcs.quantizer.pre_quant import get_named_linears
from srcs.utils.save_layer_werights import load_saved_layer
from srcs.difference.differential_encoding import (
    diff_encode_int4,
    diff_encode_uint4,
    diff_decode_int4,
    stat_diff,
    stat_diff_zp_centered,
)
from srcs.utils.run_lengths_calculate import compute_run_lengths
from srcs.utils.utils import (
    release_memory,
    save_quantized_weigths,
    save_log,
    save_json_file,
)
from srcs.utils.reorder import reorder_tile


def analyze_diff_model(model_name, out_dir, skip_if_exist=True):
    """Analyze a model: quantize weights, compute diff run-length stats, and save results."""
    group_size = 128
    strategies = [
        ("real_symm", False, None),
        # ("real_zero_point", True, None),
        # ("group_symm", False, group_size),
        # ("group_zero_point", True, group_size),
    ]
    print(f"Loading model {model_name}...")
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        config=config,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()

    for method, zero_point, gs in strategies:
        log_lines = [f"Model: {model_name} | Strategy: {method}"]
        global_rl_counter = Counter()
        layer_rl_dict = {}
        zero_ratio_dict = {}
        modules = get_named_linears(model)

        for i, (name, module) in enumerate(modules.items()):
            if i == 0:  # 只处理第一层
                weight = module.weight.data.flatten()
                quantized = real_quantize_tensor(
                    weight, zero_point=zero_point, group_size=gs
                )
                diff = quantized
                runs, len_counter = compute_run_lengths(diff)
                global_rl_counter.update(len_counter)
                layer_rl_dict[name] = dict(len_counter)
                # 零值比例
                zero_runs = [l for v, l in runs if v == 0]
                zero_elem = sum(zero_runs)
                zero_ratio = zero_elem / weight.numel()
                zero_ratio_dict[name] = zero_ratio

                line = (
                    f"Layer: {name} | Original: {weight.numel():>8} | "
                    f"Runs: {len(runs):>6} | ZeroRatio: {zero_ratio:.4f}"
                )
                print(line)
                print(f"调试第一层完成: {name}")
                break  # 处理完第一层后退出循环
        # for name, module in modules.items():
        #     weight = module.weight.data.flatten()
        #     quantized = real_quantize_tensor(
        #         weight, zero_point=zero_point, group_size=gs
        #     )
        #     # diff = diff_encode_int4(weight, tile=128)
        #     diff = quantized
        #     runs, len_counter = compute_run_lengths(diff)
        #     global_rl_counter.update(len_counter)
        #     layer_rl_dict[name] = dict(len_counter)

        #     # 零值比例
        #     zero_runs = [l for v, l in runs if v == 0]
        #     zero_elem = sum(zero_runs)
        #     zero_ratio = zero_elem / weight.numel()
        #     zero_ratio_dict[name] = zero_ratio

        #     line = (
        #         f"Layer: {name} | Original: {weight.numel():>8} | "
        #         f"Runs: {len(runs):>6} | ZeroRatio: {zero_ratio:.4f}"
        #     )
        #     print(line)
        #     log_lines.append(line)

        #     # 释放内存
        #     del weight, quantized, runs, len_counter, zero_runs, diff
        #     release_memory()


def dubug_diff_model(layer_path, index, use_diff, use_sort):
    global_rl_counter = Counter()
    layer_rl_dict = {}
    zero_ratio_dict = {}

    weight, bias, info = load_saved_layer(layer_path, index)
    name = info["layer_name"]
    print(f"Loaded layer {index} from {layer_path}: {info['layer_name']}")
    quantized = real_quantize_tensor(weight, zero_point=False, group_size=128)

    if use_sort:
        quantized = torch.sort(quantized.view(-1))[0]

    if use_diff:
        diff = diff_encode_int4(quantized, tile=128)
    else:
        diff = quantized

    cov2, cov3, same, long4 = stat_diff(diff, tile=128)

    runs, len_counter = compute_run_lengths(diff)
    global_rl_counter.update(len_counter)

    # 零值比例
    zero_runs = [l for v, l in runs if v == 0]
    zero_elem = sum(zero_runs)
    zero_ratio = zero_elem / weight.numel()

    line = (
        f"Layer: {name} | Original: {weight.numel():>8} | "
        f"Runs: {len(runs):>6} | ZeroRatio: {zero_ratio:.4f} | Use_diff: {use_diff} \n"
        f" | Cov2: {cov2:.4f} | Cov3: {cov3:.4f} | Same: {same:.4f} | Long4: {long4:.4f}"
    )
    print(line)


def test_different_quantizers(layer_path, index):
    """
    对比不同量化方案在1) 原始量化值 和 2) 差分编码后的游程统计信息.
    """
    group_size = 128
    strategies = [
        ("real_symm", False, None),
        ("real_zero_point", True, None),
        ("group_symm", False, group_size),
        ("group_zero_point", True, group_size),
    ]

    weight, bias, info = load_saved_layer(layer_path, index)
    name = info["layer_name"]
    print(f"\nLayer: {name} | Original elems: {weight.numel():>8}")

    for method, zero_point, gs in strategies:
        print("-" * 60)
        print(f"量化方法: {method}")
        quantized = real_quantize_tensor(weight, zero_point=zero_point, group_size=gs)

        # 原始量化值统计
        runs, len_counter = compute_run_lengths(quantized)
        zero_runs = [l for v, l in runs if v == 0]
        zero_ratio_orig = sum(zero_runs) / weight.numel()
        print("[原始量化值]")
        print(f"  Runs: {len(runs):>6} | ZeroRatio: {zero_ratio_orig:.4f}")

        # 差分编码后统计
        # tile = quantized.numel()
        tile = 128 * 2
        if zero_point:
            diff_encoded = diff_encode_uint4(quantized, tile=tile)
        else:
            diff_encoded = diff_encode_int4(quantized, tile=tile)

        cov2, cov3, same, long4 = stat_diff(diff_encoded, tile=tile)
        runs_diff, _ = compute_run_lengths(diff_encoded)
        zero_runs_diff = [l for v, l in runs_diff if v == 0]
        zero_ratio_diff = sum(zero_runs_diff) / weight.numel()

        print("[差分编码后]")
        print(f"  Runs: {len(runs_diff):>6} | ZeroRatio: {zero_ratio_diff:.4f}")
        print(
            f"  Cov2: {cov2:.4f} | Cov3: {cov3:.4f} | Same: {same:.4f} | Long4: {long4:.4f}"
        )

        # 重排后再差分编码统计
        w_reord, _ = reorder_tile(quantized, tile_size=128)
        if zero_point:
            diff_reord = diff_encode_uint4(w_reord, tile=tile)
        else:
            diff_reord = diff_encode_int4(w_reord, tile=tile)
        cov2_r, cov3_r, same_r, long4_r = stat_diff(diff_reord, tile=tile)
        runs_reord, _ = compute_run_lengths(diff_reord)
        zero_runs_reord = [l for v, l in runs_reord if v == 0]
        zero_ratio_reord = sum(zero_runs_reord) / weight.numel()

        print("[重排后再差分编码]")
        print(f"  Runs: {len(runs_reord):>6} | ZeroRatio: {zero_ratio_reord:.4f}")
        print(
            f"  Cov2: {cov2_r:.4f} | Cov3: {cov3_r:.4f} | Same: {same_r:.4f} | Long4: {long4_r:.4f}"
        )


def test_group_zero_point(layer_path, index):
    """基于分组 + zero-point 的量化 + 差分编码的游程统计"""
    group_size = 128
    tile = group_size

    weight, bias, info = load_saved_layer(layer_path, index)
    name = info["layer_name"]
    print(f"\nLayer: {name} | Original elems: {weight.numel():>8}")
    quantized, zero_point, scale = real_quantize_tensor(
        weight, zero_point=True, group_size=group_size, return_scale=True
    )

    # 对原始量化值分布进行统计
    cov2, cov3, same, long4 = stat_diff_zp_centered(quantized, zp=zero_point, tile=tile)
    runs, len_counter = compute_run_lengths(quantized)
    zero_runs = [l for v, l in runs if v == 0]
    zero_ratio_orig = sum(zero_runs) / weight.numel()
    print("[原始量化值, 基于zero-point进行统计]")
    print(f"  Runs: {len(runs):>6} | ZeroRatio: {zero_ratio_orig:.4f}")
    print(
        f"  Cov2: {cov2:.4f} | Cov3: {cov3:.4f} | Same: {same:.4f} | Long4: {long4:.4f}"
    )

    del cov2, cov3, same, long4
    del runs, len_counter, zero_runs

    # 差分编码后统计
    diff_encoded = diff_encode_uint4(quantized, tile=tile)
    cov2, cov3, same, long4 = stat_diff(diff_encoded, tile=tile)
    runs_diff, _ = compute_run_lengths(diff_encoded)
    zero_runs_diff = [l for v, l in runs_diff if v == 0]
    zero_ratio_diff = sum(zero_runs_diff) / weight.numel()

    print("[差分编码后]")
    print(f"  Runs: {len(runs_diff):>6} | ZeroRatio: {zero_ratio_diff:.4f}")
    print(
        f"  Cov2: {cov2:.4f} | Cov3: {cov3:.4f} | Same: {same:.4f} | Long4: {long4:.4f}"
    )


def calate_diff_distribution(layer_path, index):
    """计算差分值的分布"""
    group_size = 128
    tile = group_size

    weight, bias, info = load_saved_layer(layer_path, index)
    name = info["layer_name"]
    print(f"\nLayer: {name} | Original elems: {weight.numel():>8}")
    quantized, zero_point, scale = real_quantize_tensor(
        weight, zero_point=True, group_size=group_size, return_scale=True
    )

    diff_encoded = diff_encode_uint4(quantized, tile=tile)
    abs_diff = diff_encoded.abs().int()
    pct = torch.bincount(abs_diff, minlength=16) / abs_diff.numel()
    print("差分值分布:")
    print(f"pct[:4]: {pct[:4]}, sum: {sum(pct[:4]).item():.4f}")  # 0,1,2,3 占比


if __name__ == "__main__":
    # model_name = "facebook/opt-125m"

    layer_path = "output_weights/facebook_opt-125m_layers/"
    # layer_path = "output_weights/EleutherAI_gpt-neo-2.7B_layers/"

    # out_dir = "./output_diff/"
    # analyze_diff_model(model_name, out_dir)

    # for index in range(1):
    #     dubug_diff_model(layer_path, index, False, use_sort=False)
    #     dubug_diff_model(layer_path, index, False, use_sort=True)
    #     dubug_diff_model(layer_path, index, True, use_sort=False)
    #     dubug_diff_model(layer_path, index, True, use_sort=True)

    for index in range(0, 5):
        # test_different_quantizers(layer_path, index)
        # test_group_zero_point(layer_path, index)
        calate_diff_distribution(layer_path, index)

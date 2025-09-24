import torch
from typing import Optional
from quantizer.pre_quant import get_blocks, get_named_linears


def real_quantize_tensor(
    tensor,
    zero_point: bool = False,
    group_size: Optional[int] = None,
    rerurn_scale=False,
):
    if zero_point:
        if group_size is None:
            quantized, zero_point, scale = real_zero_point_quantize_to_4bit(tensor)
        else:
            quantized, zero_point, scale = group_zero_point_quantize_to_4bit(
                tensor, group_size
            )
        return quantized if not rerurn_scale else (quantized, zero_point, scale)

    else:
        if group_size is None:
            quantized, scale = real_symm_quantize_to_4bit(tensor)
        else:
            quantized, scale = group_symm_quantize_to_4bit(tensor, group_size)

        return quantized if not rerurn_scale else (quantized, scale)


@torch.no_grad()
def real_symm_quantize_to_4bit(tensor):
    """Symmetric INT quantization to 4 bits."""
    max_abs_val = torch.max(torch.abs(tensor))
    scale = max_abs_val / 7.0 if max_abs_val != 0 else 1.0

    quantized_int = torch.round(tensor / scale).clamp(-8, 7).to(torch.int8)
    return quantized_int, scale


@torch.no_grad()
def group_symm_quantize_to_4bit(tensor, group_size=128):
    """Symmetric grouped INT quantization to 4 bits."""
    original_shape = tensor.shape
    flattened = tensor.flatten()
    num_groups = (flattened.numel() + group_size - 1) // group_size

    quantized_data = torch.zeros_like(flattened, dtype=torch.int8)
    scales = torch.zeros(num_groups, device=tensor.device, dtype=tensor.dtype)

    for i in range(num_groups):
        start_idx = i * group_size
        end_idx = min((i + 1) * group_size, flattened.numel())
        group_data = flattened[start_idx:end_idx]

        max_abs_val = torch.max(torch.abs(group_data))
        scale = max_abs_val / 7.0 if max_abs_val != 0 else 1.0

        quantized_group = torch.round(group_data / scale).clamp(-8, 7).to(torch.int8)
        quantized_data[start_idx:end_idx] = quantized_group
        scales[i] = scale

    quantized_tensor = quantized_data.reshape(original_shape)
    return quantized_tensor, scales


@torch.no_grad()
def real_zero_point_quantize_to_4bit(tensor):
    """Zero-point quantization to 4 bits."""
    if torch.all(tensor == 0):
        scale = 1.0
        zero_point = 0
        quantized = torch.zeros_like(tensor, dtype=torch.uint8)
        return quantized, scale, zero_point
    min_val = tensor.min()
    max_val = tensor.max()
    scale = (max_val - min_val) / 15
    if scale == 0:
        scale = 1.0
    zero_point = torch.round(-min_val / scale).clamp(0, 15)
    quantized = torch.round(tensor / scale + zero_point).clamp(0, 15).to(torch.uint8)
    return quantized, zero_point, scale


@torch.no_grad()
def group_zero_point_quantize_to_4bit(tensor, group_size=128):
    """Zero point, Group quantization to 4 bits."""
    original_shape = tensor.shape
    flattened = tensor.flatten()
    num_groups = (flattened.numel() + group_size - 1) // group_size

    quantized = torch.zeros_like(flattened, dtype=torch.uint8)
    scales = torch.zeros(num_groups)
    zero_points = torch.zeros(num_groups, dtype=torch.uint8)

    for i in range(num_groups):
        start_idx = i * group_size
        end_idx = min((i + 1) * group_size, flattened.numel())
        group_tensor = flattened[start_idx:end_idx]

        min_val = group_tensor.min()
        max_val = group_tensor.max()

        if max_val - min_val < 1e-6:
            scale = 1.0
            zero_point = 0
        else:
            scale = (max_val - min_val) / 15
            zero_point = torch.round(-min_val / scale).clamp(0, 15)

        group_quantized = (
            torch.round(group_tensor / scale + zero_point).clamp(0, 15).to(torch.uint8)
        )

        quantized[start_idx:end_idx] = group_quantized
        scales[i] = scale
        zero_points[i] = zero_point

    return quantized.reshape(original_shape), zero_points, scales


if __name__ == "__main__":

    def test_edge_cases():
        """测试边界情况"""
        print("\n" + "=" * 40)
        print("测试边界情况")
        print("=" * 40)

        # 测试非常小的值
        small_values = torch.tensor([1e-6, -1e-6, 2e-6, -2e-6])
        test_tensor(small_values, "极小值")

        # 测试非常大的值
        large_values = torch.tensor([1e6, -1e6, 2e6, -2e6])
        test_tensor(large_values, "极大值")

        # 测试NaN和Inf
        print("\n测试NaN和Inf处理:")
        try:
            nan_tensor = torch.tensor([1.0, float("nan"), 3.0])
            real_symm_quantize_to_4bit(nan_tensor)
            print("警告: NaN值未被正确处理")
        except:
            print("✓ NaN值正确抛出异常")

        try:
            inf_tensor = torch.tensor([1.0, float("inf"), 3.0])
            real_symm_quantize_to_4bit(inf_tensor)
            print("警告: Inf值未被正确处理")
        except:
            print("✓ Inf值正确抛出异常")

    def test_group_boundaries():
        """测试分组边界情况"""
        print("\n测试分组边界情况:")

        # 创建刚好能被分组大小整除的张量
        perfect_fit = torch.arange(16).float()  # 16个元素,分组大小4
        print(f"完美分组张量: {perfect_fit}")

        group_quant = group_symm_quantize_to_4bit(perfect_fit, group_size=4)
        print(f"完美分组量化结果: {group_quant}")

        # 创建不能被分组大小整除的张量
        imperfect_fit = torch.arange(18).float()  # 18个元素,分组大小4
        print(f"不完美分组张量: {imperfect_fit}")

        group_quant_imperfect = group_symm_quantize_to_4bit(imperfect_fit, group_size=4)
        print(f"不完美分组量化结果: {group_quant_imperfect}")

        # 验证分组数量
        assert group_quant.numel() == 16, "完美分组数量错误"
        assert group_quant_imperfect.numel() == 18, "不完美分组数量错误"
        print("✓ 分组数量正确")

    def validate_quantization_ranges(
        symm_quant, group_symm_quant, zp_quant, group_zp_quant
    ):
        """验证量化结果的范围是否正确"""
        print("\n验证范围:")

        # 对称量化应该在[-8, 7]范围内
        if isinstance(symm_quant, torch.Tensor):
            symm_min, symm_max = symm_quant.min(), symm_quant.max()
            assert -8 <= symm_min <= 7, f"对称量化最小值错误: {symm_min}"
            assert -8 <= symm_max <= 7, f"对称量化最大值错误: {symm_max}"
            print(f"✓ 对称量化范围正确: [{symm_min}, {symm_max}]")

        # 分组对称量化应该在[-8, 7]范围内
        group_symm_min, group_symm_max = group_symm_quant.min(), group_symm_quant.max()
        assert -8 <= group_symm_min <= 7, f"分组对称量化最小值错误: {group_symm_min}"
        assert -8 <= group_symm_max <= 7, f"分组对称量化最大值错误: {group_symm_max}"
        print(f"✓ 分组对称量化范围正确: [{group_symm_min}, {group_symm_max}]")

        # Zero-point量化应该在[0, 15]范围内
        if isinstance(zp_quant, tuple):
            zp_tensor = zp_quant[0]
        else:
            zp_tensor = zp_quant
        zp_min, zp_max = zp_tensor.min(), zp_tensor.max()
        assert 0 <= zp_min <= 15, f"Zero-point量化最小值错误: {zp_min}"
        assert 0 <= zp_max <= 15, f"Zero-point量化最大值错误: {zp_max}"
        print(f"✓ Zero-point量化范围正确: [{zp_min}, {zp_max}]")

        # 分组Zero-point量化应该在[0, 15]范围内
        group_zp_min, group_zp_max = group_zp_quant.min(), group_zp_quant.max()
        assert 0 <= group_zp_min <= 15, f"分组Zero-point量化最小值错误: {group_zp_min}"
        assert 0 <= group_zp_max <= 15, f"分组Zero-point量化最大值错误: {group_zp_max}"
        print(f"✓ 分组Zero-point量化范围正确: [{group_zp_min}, {group_zp_max}]")

    def test_tensor(tensor, name):
        """测试单个张量的各种量化方法"""
        print(f"\n--- {name} ---")
        print(f"原始张量: {tensor}")
        print(f"范围: [{tensor.min():.3f}, {tensor.max():.3f}]")

        # 对称量化(无分组)
        symm_quantized = real_symm_quantize_to_4bit(tensor.clone())
        print(f"\n对称量化: {symm_quantized}")
        print(f"量化范围: [{symm_quantized.min()}, {symm_quantized.max()}]")

        # 对称分组量化
        group_symm_quantized = group_symm_quantize_to_4bit(tensor.clone(), group_size=4)
        print(f"对称分组量化: {group_symm_quantized}")
        print(
            f"分组量化范围: [{group_symm_quantized.min()}, {group_symm_quantized.max()}]"
        )

        # Zero-point量化
        zp_quantized = real_zero_point_quantize_to_4bit(tensor.clone())
        print(f"Zero-point量化: {zp_quantized}")
        if isinstance(zp_quantized, tuple):
            print(
                f"Zero-point量化范围: [{zp_quantized[0].min()}, {zp_quantized[0].max()}]"
            )
        else:
            print(f"Zero-point量化范围: [{zp_quantized.min()}, {zp_quantized.max()}]")

        # Zero-point分组量化
        group_zp_quantized = group_zero_point_quantize_to_4bit(
            tensor.clone(), group_size=4
        )
        print(f"Zero-point分组量化: {group_zp_quantized}")
        print(
            f"Zero-point分组范围: [{group_zp_quantized.min()}, {group_zp_quantized.max()}]"
        )

        # 验证值范围是否正确
        validate_quantization_ranges(
            symm_quantized, group_symm_quantized, zp_quantized, group_zp_quantized
        )

    def test_quantization_functions():
        """测试所有量化函数"""
        print("=" * 60)
        print("测试量化函数")
        print("=" * 60)

        # 测试用例1:全零张量
        print("\n1. 测试全零张量:")
        zero_tensor = torch.zeros(10)
        test_tensor(zero_tensor, "全零张量")

        # 测试用例2:全相同值
        print("\n2. 测试全相同值:")
        constant_tensor = torch.ones(10) * 3.14
        test_tensor(constant_tensor, "全相同值")

        # 测试用例3:正数范围
        print("\n3. 测试正数范围:")
        positive_tensor = torch.tensor(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        )
        test_tensor(positive_tensor, "正数范围")

        # 测试用例4:负数范围
        print("\n4. 测试负数范围:")
        negative_tensor = torch.tensor(
            [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0, -10.0]
        )
        test_tensor(negative_tensor, "负数范围")

        # 测试用例5:混合正负数
        print("\n5. 测试混合正负数:")
        mixed_tensor = torch.tensor(
            [-5.0, -2.0, 0.0, 3.0, 7.0, -8.0, 4.0, -1.0, 6.0, 2.0]
        )
        test_tensor(mixed_tensor, "混合正负数")

        # 测试用例6:大范围值
        print("\n6. 测试大范围值:")
        large_range_tensor = torch.tensor([-100.0, -50.0, 0.0, 50.0, 100.0])
        test_tensor(large_range_tensor, "大范围值")

        # 测试用例7:随机张量
        print("\n7. 测试随机张量:")
        random_tensor = torch.randn(20) * 5
        test_tensor(random_tensor, "随机张量")

        # 测试用例8:分组量化边界情况
        print("\n8. 测试分组量化边界:")
        test_group_boundaries()

    # 运行所有测试
    test_quantization_functions()
    test_edge_cases()

    print("\n" + "=" * 60)
    print("所有测试完成!")
    print("=" * 60)

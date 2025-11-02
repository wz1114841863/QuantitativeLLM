import numpy as np


class LogQuantizer:
    """
    实现一个多码本/硬件友好的对数伪量化方案.

    - 码本是数学定义的 (非k-means).
    - 为每个组选择MSE最小的最优码本.
    - 将权重映射为4-bit索引 (0-15).
    """

    def __init__(self, num_codebooks=4, log_base=1.8):
        """
        初始化量化器并生成码本.

        Args:
            num_codebooks (int): 要生成的码本数量 (S).
            log_base (float): 对数量化的底数,控制码本中值的"稀疏"程度.
        """
        self.num_codebooks = num_codebooks
        self.log_base = log_base
        self.num_levels = 7  # 7个正值 + 7个负值 + 1个零

        # 我们的S个码本,每个都是一个 (16,) 的numpy数组
        self.all_codebooks = self._generate_codebooks()

        # 将码本堆叠成一个 [S, 16] 的大矩阵,以便于并行计算
        self.codebook_matrix = np.stack(self.all_codebooks)

        # print(f"码本数量 (S): {self.num_codebooks}")
        # print(
        #     f"码本结构: 1 (零) + {self.num_levels} (正) + {self.num_levels} (负) + 1 (保留)"
        # )
        # print(f"第一个码本 (示例): \n{self.all_codebooks[0]}")

    def _generate_codebooks(self):
        """
        生成S个数学定义的对数码本.
        """
        codebooks = []

        # 我们选择S个不同的基础缩放因子 (base_scale)
        # 这些值是"超参数",您可以根据实验进行调整
        base_scales = np.logspace(
            -3, -1.5, self.num_codebooks
        )  # e.g., [0.001, 0.003, ..., 0.03]

        for base_scale in base_scales:
            # 码本有16个槽位 (index 0-15)
            codebook = np.zeros(16, dtype=np.float16)

            # Index 0: 0.0
            codebook[0] = 0.0

            # Index 1-7: 正的对数值
            positive_levels = base_scale * (self.log_base ** np.arange(self.num_levels))
            codebook[1 : self.num_levels + 1] = positive_levels

            # Index 8-14: 负的对数值
            codebook[self.num_levels + 1 : 2 * self.num_levels + 1] = -positive_levels

            # Index 15: 保留值.
            # 我们可以用它来标记"离群值",这完美契合了我们
            # "有界编码" (Bounded Coding) 的硬件设计思路.
            # 在这个纯算法实验中,我们暂时将其设为NaN,表示不使用.
            codebook[15] = np.nan

            codebooks.append(codebook)

        return codebooks

    def _find_nearest_indices(self, group_weights, codebook):
        """
        [核心映射逻辑]
        高效地为一组权重找到给定码本中的最近邻索引.
        """
        # 1. 扩展维度以利用NumPy广播
        # group_weights: (512,) -> (512, 1)
        # codebook: (16,) -> (1, 16)
        weights_expanded = np.expand_dims(group_weights, axis=1)
        codebook_expanded = np.expand_dims(codebook, axis=0)

        # 2. 计算差值
        # diffs: (512, 16)
        # 每一行代表一个权重与码本中所有16个值的差值
        diffs = np.abs(weights_expanded - codebook_expanded)

        # 3. 处理NaN (我们的保留值)
        # 我们将与NaN的差值设为无穷大,确保它永远不会被选为"最近"
        diffs[np.isnan(diffs)] = np.inf

        # 4. 找到最小差值的索引
        # np.argmin在axis=1上操作,为每个权重(每一行)返回最小值的列索引
        # indices: (512,)
        indices = np.argmin(diffs, axis=1)

        return indices.astype(np.uint8)

    def _get_mse(self, group_weights, codebook, indices):
        """
        计算使用此码本进行量化后的均方误差(MSE).
        """
        dequantized_weights = codebook[indices]
        mse = np.mean((group_weights - dequantized_weights) ** 2)
        return mse

    def quantize_group(self, group_weights):
        """
        对一个权重组进行量化.
        1. 找到最优码本 (按MSE)
        2. 返回该组的索引流和所选的码本ID
        """
        best_codebook_id = -1
        min_mse = np.inf

        cached_indices = []
        # 并行测试所有S个码本
        for i, codebook in enumerate(self.all_codebooks):
            indices = self._find_nearest_indices(group_weights, codebook)
            cached_indices.append(indices)

            mse = self._get_mse(group_weights, codebook, indices)

            if mse < min_mse:
                min_mse = mse
                best_codebook_id = i

        index_stream = cached_indices[best_codebook_id]
        return index_stream, best_codebook_id, min_mse

    def dequantize_group(self, index_stream, codebook_id):
        """
        根据索引流和码本ID, 恢复出量化后的权重.
        (用于您的困惑度评测)
        """
        if codebook_id >= len(self.all_codebooks):
            raise ValueError(f"无效的码本ID: {codebook_id}")
        codebook = self.all_codebooks[codebook_id]
        dequantized_weights = codebook[index_stream]

        return dequantized_weights


if __name__ == "__main__":

    # 1. 创建量化器实例 (这将生成4个码本)
    quantizer = LogQuantizer(num_codebooks=4)

    # 2. 模拟一个权重组 (例如 group_size=512)
    #    这个分布是LLM权重的典型特征:一个以0为中心的高斯/拉普拉斯分布
    np.random.seed(42)
    sample_group = np.random.normal(loc=0.0, scale=0.02, size=512).astype(np.float16)

    print(f"\n--- 正在量化一个示例权重组 (size={sample_group.shape}) ---")

    # 3. [离线压缩]
    #    - 找到最优码本
    #    - 生成索引流
    index_stream, best_cb_id, mse = quantizer.quantize_group(sample_group)

    print(f"最优码本ID: {best_cb_id}")
    print(f"量化后的MSE: {mse:.8f}")
    print(f"生成的索引流 (前20个): {index_stream[:20]}")
    print(
        f"索引流中'0'的占比: {np.count_nonzero(index_stream == 0) / len(index_stream):.2%}"
    )

    # (此时, 您会将 index_stream 送入Golomb-Rice编码器,
    #  并存储 best_cb_id 作为元数据)

    # 4. [在线解压]
    #    (模拟硬件解码器的工作: GR解码器解码出index_stream,
    #     然后PDU的最后一级使用best_cb_id进行查表)
    dequantized_weights = quantizer.dequantize_group(index_stream, best_cb_id)

    print(f"\n--- 正在反量化 ---")
    print(f"原始权重 (前5个): {sample_group[:5]}")
    print(f"恢复权重 (前5个): {dequantized_weights[:5]}")

    # 5. [实验验证]
    #    验证恢复后的权重与原始权重的MSE是否一致
    validation_mse = np.mean((sample_group - dequantized_weights) ** 2)
    print(f"恢复后的MSE (验证): {validation_mse:.8f}")

    assert np.isclose(mse, validation_mse), "MSE不匹配, 逻辑有误!"
    print("\n--- A/B测试平台验证成功 ---")

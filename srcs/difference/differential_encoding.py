import torch


def diff_encode_int4(W, tile=128):
    """Differential encoding for INT4 weights."""
    W = W.view(-1, tile)
    W_diff = torch.zeros_like(W)
    W_diff[:, 0] = W[:, 0]
    W_diff[:, 1:] = W[:, 1:] - W[:, :-1]
    W_diff = torch.round(W_diff).clamp(-8, 7).to(torch.int8)
    return W_diff.view(-1)


def diff_decode_int4(W_diff, tile=128):
    """Decode differential encoded INT4 weights."""
    W_diff = W_diff.view(-1, tile)
    W = torch.zeros_like(W_diff)
    W[:, 0] = W_diff[:, 0]
    for i in range(1, W.shape[1]):
        W[:, i] = W[:, i - 1] + W_diff[:, i].float()
    return W.view(-1)


def stat_diff(W_diff, tile=128):
    W_diff = W_diff.view(-1, tile)
    # 符号覆盖率
    cov2 = (W_diff.abs() <= 1).float().mean().item()
    cov3 = (W_diff.abs() <= 3).float().mean().item()
    same = (W_diff == 0).float().mean().item()
    # 游程统计
    long4 = 0.0
    for row in W_diff:
        _, runlen = torch.unique_consecutive(row, return_counts=True)
        long4 += (runlen >= 4).sum().item()
    long4 /= W_diff.numel()
    return cov2, cov3, same, long4

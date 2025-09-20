import torch


def diff_encode_int4(W, tile=64):
    """Differential encoding for INT4 weights."""
    W = W.view(-1, tile)
    W_diff = torch.zeros_like(W)
    W_diff[:, 0] = W[:, 0]
    W_diff[:, 1:] = W[:, 1:] - W[:, :-1]
    W_diff = torch.round(W_diff).clamp(-8, 7).to(torch.int8)
    return W_diff.view_as(-1)

def diff_decode_int4(W_diff, tile=64):
    """Decode differential encoded INT4 weights."""
    W_diff = W_diff.view(-1, tile)
    W = torch.zeros_like(W_diff)
    W[:, 0] = W_diff[:, 0]
    for i in range(1, W.shape[1]):
        W[:, i] = W[:, i - 1] + W_diff[:, i].float()
    return W.view_as(-1)

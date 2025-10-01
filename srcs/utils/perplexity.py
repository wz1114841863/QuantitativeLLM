import torch
import random

from tqdm import tqdm
from datasets import load_dataset


def calc_perplexity_wikitext(model, tokenizer):
    """计算模型在wikitext-2-raw-v1数据集上的困惑度"""
    model.seqlen = 2048
    model = model.eval()

    testenc = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    testenc = tokenizer("\n\n".join(testenc["text"]), return_tensors="pt")
    testenc = testenc.input_ids.to(model.device)
    nsamples = testenc.numel() // model.seqlen
    loss_fct = torch.nn.CrossEntropyLoss()

    nlls = []
    for i in tqdm(range(nsamples), desc="evaluating..."):
        batch = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)].to(
            model.device
        )
        with torch.no_grad():
            lm_logits = model(batch).logits
        shift_logits = lm_logits[:, :-1, :].contiguous().float()
        shift_labels = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)][:, 1:]
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(f"wikitext perplexity: {ppl.item()}")
    return ppl.item()


def calc_perplexity_c4(model, tokenizer, seqlen=2048, num_samples=256, device=None):
    """计算模型在c4数据集上的困惑度(perplexity)"""
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    random.seed(0)

    if device is None:
        device = (
            model.device
            if hasattr(model, "device")
            else "cuda" if torch.cuda.is_available() else "cpu"
        )
    model = model.to(device).eval()

    print("Loading C4 dataset...")
    test_dataset = load_dataset(
        "allenai/c4",
        data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
        split="validation",
    )

    # 准备测试数据
    print("Preparing test data...")
    valenc = []
    attempts = 0
    max_attempts = num_samples * 10  # 防止无限循环
    while len(valenc) < num_samples and attempts < max_attempts:
        i = random.randint(0, len(test_dataset) - 1)
        text = test_dataset[i]["text"]

        # Tokenize文本
        encoded = tokenizer(text, return_tensors="pt", truncation=False)
        input_ids = encoded.input_ids

        # 确保文本足够长
        if input_ids.shape[1] > (seqlen + 1):
            start_idx = random.randint(0, input_ids.shape[1] - seqlen - 1)
            end_idx = start_idx + seqlen
            segment = input_ids[:, start_idx:end_idx]
            valenc.append(segment)

        attempts += 1

    if len(valenc) < num_samples:
        print(
            f"Warning: Only found {len(valenc)} suitable samples out of {num_samples} requested"
        )

    # 合并所有样本
    testenc = (
        torch.cat(valenc, dim=1) if valenc else torch.empty(1, 0, dtype=torch.long)
    )

    if testenc.numel() == 0:
        raise ValueError("No suitable samples found for perplexity calculation")

    # 计算困惑度
    nsamples = testenc.shape[1] // seqlen
    if nsamples == 0:
        raise ValueError("Not enough data for perplexity calculation")

    loss_fct = torch.nn.CrossEntropyLoss()
    nlls = []

    print(f"Calculating perplexity over {nsamples} sequences...")
    with tqdm(range(nsamples), desc="Evaluating perplexity") as progress:
        for i in progress:
            # 获取当前batch
            start_idx = i * seqlen
            end_idx = (i + 1) * seqlen
            batch = testenc[:, start_idx:end_idx].to(device)

            # 前向传播
            with torch.no_grad():
                try:
                    # 尝试不同的输出格式
                    if hasattr(model, "forward"):
                        outputs = model.forward(batch)
                    else:
                        outputs = model(batch)

                    # 处理不同的输出格式
                    if isinstance(outputs, torch.Tensor):
                        lm_logits = outputs
                    elif isinstance(outputs, tuple):
                        lm_logits = outputs[0]
                    elif hasattr(outputs, "logits"):
                        lm_logits = outputs.logits
                    else:
                        raise ValueError("Unable to extract logits from model output")

                except Exception as e:
                    print(f"Error during forward pass: {e}")
                    continue

            # 计算损失
            shift_logits = lm_logits[:, :-1, :].contiguous().float()
            shift_labels = batch[:, 1:].contiguous().to(device)

            # 确保形状匹配
            if shift_logits.shape[1] != shift_labels.shape[1]:
                min_len = min(shift_logits.shape[1], shift_labels.shape[1])
                shift_logits = shift_logits[:, :min_len, :]
                shift_labels = shift_labels[:, :min_len]

            loss = loss_fct(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
            )

            # 累加负对数似然
            neg_log_likelihood = loss * seqlen
            nlls.append(neg_log_likelihood.item())

            # 更新进度条
            progress.set_postfix(
                {"current_ppl": torch.exp(torch.tensor(nlls).mean()).item()}
            )

    # 计算最终困惑度
    total_nll = torch.tensor(nlls).sum()
    total_tokens = nsamples * seqlen
    ppl = torch.exp(total_nll / total_tokens)

    print(f"C4 perplexity: {ppl:.4f}")
    return ppl.item()

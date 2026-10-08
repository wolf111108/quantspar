import math
import torch
import torch.nn as nn
from datasets import load_dataset
from tqdm import tqdm


def collect_tokens_from_fineweb(
    tokenizer,
    dataset_name="HuggingFaceFW/fineweb",
    dataset_config="sample-10BT",
    split="train",
    max_eval_tokens=262144,
    min_doc_tokens=512,
):
    ds = load_dataset(
        dataset_name,
        name=dataset_config,
        split=split,
        streaming=True,
    )

    token_ids = []
    eos = tokenizer.eos_token_id

    for ex in ds:
        text = ex.get("text", "")
        if not text:
            continue

        # FineWeb token_count 是 GPT-2 tokenizer 下的数量，只做粗筛。
        if ex.get("token_count", min_doc_tokens) < min_doc_tokens:
            continue

        ids = tokenizer(text, add_special_tokens=False).input_ids
        if len(ids) < min_doc_tokens:
            continue

        token_ids.extend(ids)
        if eos is not None:
            token_ids.append(eos)

        if len(token_ids) >= max_eval_tokens:
            token_ids = token_ids[:max_eval_tokens]
            break

    if len(token_ids) == 0:
        raise RuntimeError("No FineWeb tokens collected.")

    return torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)


@torch.no_grad()
def evaluate_fineweb_ppl(
    model,
    tokenizer,
    seq_length=4096,
    max_eval_tokens=262144,
    device="cuda",
    dataset_config="sample-10BT",
):
    model.eval()

    input_ids = collect_tokens_from_fineweb(
        tokenizer=tokenizer,
        dataset_config=dataset_config,
        max_eval_tokens=max_eval_tokens,
    ).to(device)

    nsamples = input_ids.numel() // seq_length
    input_ids = input_ids[:, : nsamples * seq_length]

    loss_fct = nn.CrossEntropyLoss()
    total_nll = torch.zeros([], dtype=torch.float64, device=device)
    total_tokens = 0

    print(f"FineWeb PPL: seq_length={seq_length}, nsamples={nsamples}")

    for i in tqdm(range(nsamples), desc="FineWeb PPL"):
        batch = input_ids[:, i * seq_length : (i + 1) * seq_length]

        logits = model(batch).logits
        shift_logits = logits[:, :-1, :].contiguous().float()
        shift_labels = batch[:, 1:].contiguous()

        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

        ntok = shift_labels.numel()
        total_nll += loss.double() * ntok
        total_tokens += ntok

    ppl = torch.exp(total_nll / total_tokens).item()
    return ppl
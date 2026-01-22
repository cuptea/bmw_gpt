import concurrent.futures
import os
import re
from pathlib import Path
from typing import List, Tuple

import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm


def _process_single_file(path: Path) -> str:
    """Helper function to read and clean a single text file."""
    try:
        with path.open("r", encoding="utf-8") as f:
            content = f.read().strip()

            # Remove separator lines (lines containing only ====, ----, or similar patterns)
            lines = content.split("\n")
            filtered_lines = []
            for line in lines:
                stripped = line.strip()
                # Skip lines that are only separator characters (=, -, _, or spaces)
                if stripped and not re.fullmatch(r"[=_\- ]*", stripped):
                    filtered_lines.append(line)
            content = "\n".join(filtered_lines).strip()

            if content:  # Only return non-empty content
                return content
            return ""
    except Exception as e:
        print(f"Warning: Failed to read {path}: {e}")
        return ""


def load_text_files(data_dir: Path, sample_fraction: float) -> List[str]:
    """Load all .txt files from a directory into a list of strings using parallel processing."""
    txt_files = sorted(data_dir.glob("*.txt"))
    if not txt_files:
        raise ValueError(f"No .txt files found in {data_dir}")

    if sample_fraction < 1.0:
        keep_count = max(1, int(len(txt_files) * sample_fraction))
        txt_files = txt_files[:keep_count]

    texts: List[str] = []
    total_files = len(txt_files)
    print(f"Found {total_files} .txt files")

    # Use ThreadPoolExecutor for I/O-bound tasks (reading files)
    # Adjust max_workers based on your system's capabilities and nature of task
    # For I/O bound, more workers might be beneficial.
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() * 2) as executor:
        # Map _process_single_file to all txt_files and show progress with tqdm
        for content in tqdm(
            executor.map(_process_single_file, txt_files),
            total=total_files,
            desc="Loading files",
        ):
            if content:
                texts.append(content)

    total_chars = sum(len(text) for text in texts)
    print(f"Successfully loaded {len(texts)} text files (~{total_chars:,} characters)")
    return texts


def evaluation(model: nn.Module, valid_loader, device: str) -> float:
    # fine tune the model with BMW data
    model.eval()
    batch_loss = []

    epoch_number = 1
    for _ in tqdm(range(valid_loader.batch_per_epoch * epoch_number)):
        x, y = valid_loader.next_batch()
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        batch_loss.append(loss.item())

    return sum(batch_loss) / len(batch_loss)


def sample_text(
    model: nn.Module,
    num_return_sequences: int,
    max_length: int,
    enc,
    tokens: torch.Tensor,
    device: str,
) -> None:
    # generate! right now x is (B, T) where B = 5, T = 8
    # set the seed to 42
    x = tokens.to(device)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    while x.size(1) < max_length:
        # forward the model to get the logits
        with torch.no_grad():
            logits, _ = model(x)  # (B, T, vocab_size)
            # take the logits at the last position
            logits = logits[:, -1, :]  # (B, vocab_size)
            # get the probabilities
            probs = F.softmax(logits, dim=-1)
            # do top-k sampling of 50 (huggingface pipeline default)
            # topk_probs here becomes (5, 50), topk_indices is (5, 50)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            # select a token from the top-k probabilities
            # note: multinomial does not demand the input to sum to 1
            ix = torch.multinomial(topk_probs, 1)  # (B, 1)
            # gather the corresponding indices
            xcol = torch.gather(topk_indices, -1, ix)  # (B, 1)
            # append to the sequence
            x = torch.cat((x, xcol), dim=1)

    # print the generated text
    for i in range(num_return_sequences):
        tokens = x[i, :max_length].tolist()
        decoded = enc.decode(tokens)
        print(">", decoded)


def render_example(example) -> Tuple[dict, torch.Tensor, torch.Tensor, int]:
    """
    Given the example as a dictionary, render it as three torch tensors:
    - tokens (the tokens of context + completion, of size 4xN, as there are always 4 candidates)
    - mask (is 1 in the region of the candidate completion, where we evaluate likelihoods)
    - label (the index of the correct completion, which we hope has the highest likelihood)
    """
    enc = tiktoken.get_encoding("gpt2")
    ctx = example["question_context"]
    label = example["correct_label_index"]
    endings = [
        example["choice_1"],
        example["choice_2"],
        example["choice_3"],
        example["choice_4"],
    ]

    # data needed to reproduce this eval on the C size
    data = {
        "label": label,
        "ctx_tokens": None,
        "ending_tokens": [],
    }

    # gather up all the tokens
    ctx_tokens = enc.encode(ctx)
    data["ctx_tokens"] = ctx_tokens
    tok_rows = []
    mask_rows = []
    for end in endings:
        end_tokens = enc.encode(" " + end)  # note: prepending " " because GPT-2 tokenizer
        tok_rows.append(ctx_tokens + end_tokens)
        mask_rows.append([0] * len(ctx_tokens) + [1] * len(end_tokens))
        data["ending_tokens"].append(end_tokens)

    # have to be careful during the collation because the number of tokens in each row can differ
    max_len = max(len(row) for row in tok_rows)
    tokens = torch.zeros((4, max_len), dtype=torch.long)
    mask = torch.zeros((4, max_len), dtype=torch.long)
    for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
        tokens[i, : len(tok_row)] = torch.tensor(tok_row)
        mask[i, : len(mask_row)] = torch.tensor(mask_row)

    return data, tokens, mask, label


def iterate_examples(data):
    # there are 100 examples in total in val
    for example in data:
        yield example


@torch.no_grad()
def evaluate_multi_choice(model: nn.Module, device: str, bmw_multi_choice_data) -> float:
    model.to(device)
    num_correct_norm = 0
    num_correct = 0
    num_total = 0
    for example in iterate_examples(bmw_multi_choice_data):
        data, tokens, mask, label = render_example(example)
        tokens = tokens.to(device)
        mask = mask.to(device)

        # get the logits
        logits, _ = model(tokens)
        # evaluate the autoregressive loss at all positions
        shift_logits = (logits[..., :-1, :]).contiguous()
        shift_tokens = (tokens[..., 1:]).contiguous()
        flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_shift_tokens = shift_tokens.view(-1)
        shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction="none")
        shift_losses = shift_losses.view(tokens.size(0), -1)
        # now get the average loss just for the completion region (where mask == 1), in each row
        shift_mask = (mask[..., 1:]).contiguous()  # we must shift mask, so we start at the last prompt token
        masked_shift_losses = shift_losses * shift_mask
        # sum and divide by the number of 1s in the mask
        sum_loss = masked_shift_losses.sum(dim=1)
        avg_loss = sum_loss / shift_mask.sum(dim=1)
        # now we have a loss for each of the 4 completions
        # the one with the lowest loss should be the most likely
        pred = sum_loss.argmin().item()
        pred_norm = avg_loss.argmin().item()

        # accumulate stats
        num_total += 1
        num_correct += int(pred == label)
        num_correct_norm += int(pred_norm == label)

        # debug: pretty print a few examples, and the losses in each case
        if num_total == 10:
            print("---")
            print(f"Context:\n {example['question_context']}")
            print("Endings:")
            for i, end in enumerate(
                [
                    example["choice_1"],
                    example["choice_2"],
                    example["choice_3"],
                    example["choice_4"],
                ]
            ):
                print(f"{i} (loss: {avg_loss[i].item():.4f}) {end}")
            print(f"predicted: {pred_norm}, actual: {label}")

    print(f"{num_total} acc_norm: {num_correct_norm}/{num_total}={num_correct_norm/num_total:.4f}")
    final_acc = num_correct_norm / num_total
    return final_acc


def train(train_loader, model: nn.Module, epoch_number: int, device: str):
    # fine tune the model with BMW data
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    epoch_loss = []
    for _ in tqdm(range(epoch_number)):
        batch_loss = []
        for _ in range(train_loader.batch_per_epoch):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, loss = model(x, y)
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())

        epoch_loss.append(sum(batch_loss) / len(batch_loss))
    return model, epoch_loss

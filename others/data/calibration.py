"""
Calibration Data Loader.

Supports regular Hugging Face datasets and streaming datasets such as FineWeb.
"""

import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import AutoTokenizer
from typing import Optional, Dict, Any, List
import random


class CalibrationDataset(Dataset):
    """
    Dataset for calibration.

    For normal HF datasets, load the split and sample/select examples.
    For streaming datasets, collect num_samples examples into memory first.
    """

    def __init__(
        self,
        dataset_name: str,
        dataset_config: Optional[str],
        split: str,
        tokenizer,
        seq_length: int = 2048,
        num_samples: Optional[int] = None,
        seed: int = 42,
        text_column: str = "text",
        streaming: bool = False,
        min_text_tokens: int = 32,
    ):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.text_column = text_column
        self.streaming = streaming
        self.min_text_tokens = min_text_tokens

        print(
            f"Loading dataset: {dataset_name} "
            f"config={dataset_config}, split={split}, streaming={streaming}"
        )

        if dataset_config is None:
            raw_dataset = load_dataset(
                dataset_name,
                split=split,
                streaming=streaming,
            )
        else:
            raw_dataset = load_dataset(
                dataset_name,
                dataset_config,
                split=split,
                streaming=streaming,
            )

        if streaming:
            self.samples = self._collect_streaming_samples(
                raw_dataset,
                num_samples=num_samples,
                seed=seed,
            )
        else:
            if num_samples is not None and num_samples < len(raw_dataset):
                random.seed(seed)
                indices = random.sample(range(len(raw_dataset)), num_samples)
                raw_dataset = raw_dataset.select(indices)

            self.samples = []
            for ex in raw_dataset:
                text = self._get_text(ex)
                if text is not None:
                    self.samples.append(text)

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No valid calibration samples found. "
                f"Check dataset={dataset_name}, config={dataset_config}, "
                f"split={split}, text_column={text_column}."
            )

        print(f"Calibration samples collected: {len(self.samples)}")

    def _get_text(self, example: Dict[str, Any]) -> Optional[str]:
        if self.text_column not in example:
            raise KeyError(
                f"text_column='{self.text_column}' not found. "
                f"Available columns: {list(example.keys())}"
            )

        text = example[self.text_column]
        if text is None:
            return None

        if isinstance(text, list):
            text = "\n\n".join(str(x) for x in text)

        text = str(text).strip()
        if len(text) == 0:
            return None

        return text

    def _collect_streaming_samples(self, raw_dataset, num_samples, seed):
        # Streaming dataset cannot use len(), select(), or random indexing.
        # We optionally shuffle the stream, then take valid examples.
        if seed is not None:
            try:
                raw_dataset = raw_dataset.shuffle(seed=seed, buffer_size=10_000)
            except Exception as e:
                print(f"Warning: streaming shuffle failed: {e}")

        target = num_samples if num_samples is not None else 128
        samples = []

        for ex in raw_dataset:
            text = self._get_text(ex)
            if text is None:
                continue

            # Token-level light filter. This avoids calibrating on tiny docs.
            # 注意: 计数用的 max_length 必须取 max(seq_length, min_text_tokens)。
            # 若直接用 seq_length 截断后再与 min_text_tokens 比较, 则当
            # min_text_tokens > seq_length (如 OPT: 4096 > 2048) 时
            # len(ids) < min_text_tokens 恒成立, 过滤器永假, 会导致
            # "No valid calibration samples found"。
            count_cap = max(self.seq_length, self.min_text_tokens)
            ids = self.tokenizer(
                text,
                add_special_tokens=False,
                truncation=True,
                max_length=count_cap,
            ).input_ids

            if len(ids) < self.min_text_tokens:
                continue

            samples.append(text)

            if len(samples) >= target:
                break

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.samples[idx]

        encoded = self.tokenizer(
            text,
            max_length=self.seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }


class CalibrationDataLoader:
    """
    Data loader for calibration.
    """

    def __init__(
        self,
        dataset_name: str,
        dataset_config: Optional[str],
        split: str,
        model_name_or_path: str,
        seq_length: int = 2048,
        batch_size: int = 1,
        num_samples: Optional[int] = None,
        seed: int = 42,
        text_column: str = "text",
        streaming: bool = False,
        min_text_tokens: int = 32,
    ):
        print(f"Loading tokenizer from: {model_name_or_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.dataset = CalibrationDataset(
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            split=split,
            tokenizer=self.tokenizer,
            seq_length=seq_length,
            num_samples=num_samples,
            seed=seed,
            text_column=text_column,
            streaming=streaming,
            min_text_tokens=min_text_tokens,
        )

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

    def __iter__(self):
        return iter(self.dataloader)

    def __len__(self) -> int:
        return len(self.dataloader)


def prepare_calibration_data(
    config: Dict[str, Any],
    model_path: str,
) -> CalibrationDataLoader:
    calib_config = config["calibration"]

    return CalibrationDataLoader(
        dataset_name=calib_config["dataset"],
        dataset_config=calib_config.get("dataset_config", None),
        split=calib_config.get("split", "train"),
        model_name_or_path=model_path,
        seq_length=calib_config.get("seq_length", 2048),
        batch_size=calib_config.get("batch_size", 1),
        num_samples=calib_config.get("num_samples", None),
        seed=calib_config.get("seed", 42),
        text_column=calib_config.get("text_column", "text"),
        streaming=calib_config.get("streaming", False),
        min_text_tokens=calib_config.get("min_text_tokens", 32),
    )
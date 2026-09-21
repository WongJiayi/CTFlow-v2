import os
import random

import pandas as pd
import torch
from torch.utils.data import Dataset

P_SEX        = 0.8
P_AGE        = 0.8
P_FINDINGS   = 0.75
P_IMPRESSION = 0.9


def _build_text(row_rep, row_meta):
    parts = []
    sex = str(row_meta.get("PatientSex", "") or "").strip()
    if sex and random.random() < P_SEX:
        parts.append(f"[Sex: {sex}]")
    age = str(row_meta.get("PatientAge", "") or "").strip()
    if age and random.random() < P_AGE:
        parts.append(f"[Age: {age}]")
    findings = str(row_rep.get("Findings_EN", "") or "").strip()
    if findings and random.random() < P_FINDINGS:
        parts.append(f"Findings: {findings}")
    impression = str(row_rep.get("Impressions_EN", "") or "").strip()
    if impression and random.random() < P_IMPRESSION:
        parts.append(f"Impression: {impression}")
    if not parts:
        parts.append(findings or impression or "Normal CT scan.")
    return " ".join(parts)


class LatentBlockDataset(Dataset):
    def __init__(self, root_dir, embedding_dir, block_size=16, black_block_prob=0.3):
        self.root_dir = root_dir
        self.embedding_dir = embedding_dir
        self.block_size = block_size
        self.black_block_prob = black_block_prob

        all_paths = sorted([
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.endswith(".pt")
        ])

        # only keep samples that have a matching embedding
        self.file_paths = []
        self.embedding_paths = []
        for p in all_paths:
            emb = os.path.join(embedding_dir, os.path.basename(p))
            if os.path.exists(emb):
                self.file_paths.append(p)
                self.embedding_paths.append(emb)

        print(f"[LatentBlockDataset] {len(self.file_paths)} valid samples "
              f"(skipped {len(all_paths) - len(self.file_paths)} missing embeddings)")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        for _ in range(10):
            try:
                latent_path = self.file_paths[idx]
                embed_path = self.embedding_paths[idx]
                latent = torch.load(latent_path, map_location="cpu")  # [C, T, H, W]
                embedding = torch.load(embed_path, map_location="cpu")
                break
            except Exception:
                idx = random.randint(0, len(self.file_paths) - 1)
        else:
            raise RuntimeError(f"Failed to load a valid sample after 10 retries")
        embedding = embedding[0].unsqueeze(0)
        embedding = embedding / (embedding.norm(p=2) + 1e-6) * 1

        C, T, H, W = latent.shape

        # prepend a black block (zero block as start token)
        black_block = torch.zeros(C, self.block_size, H, W, dtype=latent.dtype)
        latent = torch.cat([black_block, latent], dim=1)  # [C, T+block_size, H, W]

        _, T_total, _, _ = latent.shape
        max_start = T_total - 2 * self.block_size

        if random.random() < self.black_block_prob:
            t = 0
        else:
            t = random.randint(0, max_start)

        block_curr = latent[:, t:t + self.block_size]
        block_next = latent[:, t + self.block_size:t + 2 * self.block_size]

        return {
            "image": block_curr,
            "video": block_next,
            "embedding": embedding,
        }


class LatentBlockDatasetV2(Dataset):
    """On-the-fly text encoding version.

    Returns raw text strings instead of pre-computed embeddings.
    The training loop encodes them with CTCLIPTextEncoder each step.

    Config params:
      root_dir      — directory of latent .pt files
      reports_csv   — path to train_reports.csv  (VolumeName, Findings_EN, Impressions_EN)
      metadata_csv  — path to train_metadata.csv (VolumeName, PatientSex, PatientAge)
      block_size    — AR block size (default 16)
      black_block_prob — prob of sampling the start (black) block (default 0.3)
    """

    def __init__(self, root_dir, reports_csv, metadata_csv,
                 block_size=16, black_block_prob=0.3):
        self.block_size = block_size
        self.black_block_prob = black_block_prob

        rep_df  = pd.read_csv(reports_csv).set_index("VolumeName")
        meta_df = pd.read_csv(metadata_csv).set_index("VolumeName")
        # deduplicate in case of multiple rows per volume
        rep_df  = rep_df[~rep_df.index.duplicated(keep="first")]
        meta_df = meta_df[~meta_df.index.duplicated(keep="first")]
        self.rep_df  = rep_df
        self.meta_df = meta_df

        self.file_paths = sorted([
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.endswith(".pt")
        ])
        print(f"[LatentBlockDatasetV2] {len(self.file_paths)} samples")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        for _ in range(10):
            try:
                latent_path = self.file_paths[idx]
                latent = torch.load(latent_path, map_location="cpu")  # [C, T, H, W]
                break
            except Exception:
                idx = random.randint(0, len(self.file_paths) - 1)
        else:
            raise RuntimeError("Failed to load a valid sample after 10 retries")

        vol_name = os.path.basename(latent_path).replace(".pt", ".nii.gz")
        row_rep  = self.rep_df.loc[vol_name].to_dict()  if vol_name in self.rep_df.index  else {}
        row_meta = self.meta_df.loc[vol_name].to_dict() if vol_name in self.meta_df.index else {}
        text = _build_text(row_rep, row_meta)

        C, T, H, W = latent.shape
        black_block = torch.zeros(C, self.block_size, H, W, dtype=latent.dtype)
        latent = torch.cat([black_block, latent], dim=1)

        _, T_total, _, _ = latent.shape
        max_start = T_total - 2 * self.block_size

        t = 0 if random.random() < self.black_block_prob else random.randint(0, max_start)

        return {
            "image": latent[:, t:t + self.block_size],
            "video": latent[:, t + self.block_size:t + 2 * self.block_size],
            "text":  text,
        }


def instantiate_dataset(configs, split=None):
    datasets = []
    for cfg in configs:
        if not cfg.get("active", False):
            continue
        name = cfg.name
        params = dict(cfg.params)

        if name == "LatentBlock":
            dataset = LatentBlockDataset(**params)
        elif name == "LatentBlockV2":
            dataset = LatentBlockDatasetV2(**params)
        else:
            raise ValueError(f"Unknown dataset name: {name}")
        datasets.append(dataset)

    if len(datasets) == 1:
        return datasets[0]
    from torch.utils.data import ConcatDataset
    return ConcatDataset(datasets)

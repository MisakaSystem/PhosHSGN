from typing import List, Tuple, Optional, Dict, NamedTuple, Union, Callable
import itertools
import os
import string
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.distance import squareform, pdist, cdist
from Bio import SeqIO
import biotite.structure as bs
from biotite.structure.io.pdb import PDBFile, get_structure
from biotite.database import rcsb
from tqdm import tqdm
import pandas as pd
import esm
import random
import tempfile
import string
torch.set_grad_enabled(False)
# This is an efficient way to delete lowercase characters and insertion characters from a string
deletekeys = dict.fromkeys(string.ascii_lowercase)
deletekeys["."] = None
deletekeys["*"] = None
translation = str.maketrans(deletekeys)

def read_sequence(filename: str) -> Tuple[str, str]:
    """ Reads the first (reference) sequences from a fasta or MSA file."""
    record = next(SeqIO.parse(filename, "fasta"))
    return record.description, str(record.seq)

def remove_insertions(sequence: str) -> str:
    """ Removes any insertions into the sequence. Needed to load aligned sequences in an MSA. """
    return sequence.translate(translation)

def read_msa(filename: str) -> List[Tuple[str, str]]:
    """ Reads the sequences from an MSA file, automatically removes insertions."""
    return [(record.description, remove_insertions(str(record.seq))) for record in SeqIO.parse(filename, "fasta")]
def extend(a, b, c, L, A, D):
    """
    input:  3 coords (a,b,c), (L)ength, (A)ngle, and (D)ihedral
    output: 4th coord
    """

    def normalize(x):
        return x / np.linalg.norm(x, ord=2, axis=-1, keepdims=True)

    bc = normalize(b - c)
    n = normalize(np.cross(b - a, bc))
    m = [bc, np.cross(n, bc), n]
    d = [L * np.cos(A), L * np.sin(A) * np.cos(D), -L * np.sin(A) * np.sin(D)]
    return c + sum([m * d for m, d in zip(m, d)])


def contacts_from_pdb(
    structure: bs.AtomArray,
    distance_threshold: float = 8.0,
    chain: Optional[str] = None,
) -> np.ndarray:
    mask = ~structure.hetero
    if chain is not None:
        mask &= structure.chain_id == chain

    N = structure.coord[mask & (structure.atom_name == "N")]
    CA = structure.coord[mask & (structure.atom_name == "CA")]
    C = structure.coord[mask & (structure.atom_name == "C")]

    Cbeta = extend(C, N, CA, 1.522, 1.927, -2.143)
    dist = squareform(pdist(Cbeta))
    
    contacts = dist < distance_threshold
    contacts = contacts.astype(np.int64)
    contacts[np.isnan(dist)] = -1
    return contacts

async def get_cmap(pdb_path,name):
    try:
        # Check if cmap exists
        cmap_tag = os.path.exists(f"./out/cmap/{name}.npy")
        if cmap_tag == False:
            pdbfile=PDBFile.read(pdb_path)
            try:
                structure=get_structure(pdbfile)[0]
            except ValueError:
                return {"err_code": -1, "err_desc": "failed to fetch cmap, return pdb path", "result": pdb_path}
            contacts=contacts_from_pdb(structure)
            print(type(contacts))

            print(f"got cmap:{contacts.shape}")
            npmascmap=contacts
            current_directory = os.getcwd()

            absolute_path = os.path.join(current_directory, f"./out/cmap/{name}.npy")
            print("cmap path:", absolute_path)
            np.save(f"./out/cmap/{name}.npy",npmascmap)
            if os.path.exists(absolute_path):
                return {"err_code": 0, "err_desc": "got cmap, return pdb path", "result": absolute_path}
            else:
                return {"err_code": -1, "err_desc": "failed to fetch cmap, return pdb path", "result": pdb_path}
        else:
            current_directory = os.getcwd()
            absolute_path = os.path.join(current_directory, f"./out/cmap/{name}.npy")
            print("cmap path:", absolute_path)
            return {"err_code": 0, "err_desc": "got cmap, return pdb path", "result": absolute_path}
    except RuntimeError as e:
        return {"err_code": -1, "err_desc": f"{e}", "result": None}
def compute_precisions(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    src_lengths: Optional[torch.Tensor] = None,
    minsep: int = 6,
    maxsep: Optional[int] = None,
    override_length: Optional[int] = None,  # for casp
):
    if isinstance(predictions, np.ndarray):
        predictions = torch.from_numpy(predictions)
    if isinstance(targets, np.ndarray):
        targets = torch.from_numpy(targets)
    if predictions.dim() == 2:
        predictions = predictions.unsqueeze(0)
    if targets.dim() == 2:
        targets = targets.unsqueeze(0)
    override_length = (targets[0, 0] >= 0).sum()

    # Check sizes
    if predictions.size() != targets.size():
        raise ValueError(
            f"Size mismatch. Received predictions of size {predictions.size()}, "
            f"targets of size {targets.size()}"
        )
    device = predictions.device

    batch_size, seqlen, _ = predictions.size()
    seqlen_range = torch.arange(seqlen, device=device)

    sep = seqlen_range.unsqueeze(0) - seqlen_range.unsqueeze(1)
    sep = sep.unsqueeze(0)
    valid_mask = sep >= minsep
    valid_mask = valid_mask & (targets >= 0)  # negative targets are invalid

    if maxsep is not None:
        valid_mask &= sep < maxsep

    if src_lengths is not None:
        valid = seqlen_range.unsqueeze(0) < src_lengths.unsqueeze(1)
        valid_mask &= valid.unsqueeze(1) & valid.unsqueeze(2)
    else:
        src_lengths = torch.full([batch_size], seqlen, device=device, dtype=torch.long)

    predictions = predictions.masked_fill(~valid_mask, float("-inf"))

    x_ind, y_ind = np.triu_indices(seqlen, minsep)
    predictions_upper = predictions[:, x_ind, y_ind]
    targets_upper = targets[:, x_ind, y_ind]

    topk = seqlen if override_length is None else max(seqlen, override_length)
    indices = predictions_upper.argsort(dim=-1, descending=True)[:, :topk]
    topk_targets = targets_upper[torch.arange(batch_size).unsqueeze(1), indices]
    if topk_targets.size(1) < topk:
        topk_targets = F.pad(topk_targets, [0, topk - topk_targets.size(1)])

    cumulative_dist = topk_targets.type_as(predictions).cumsum(-1)

    gather_lengths = src_lengths.unsqueeze(1)
    if override_length is not None:
        gather_lengths = override_length * torch.ones_like(
            gather_lengths, device=device
        )

    gather_indices = (
        torch.arange(0.1, 1.1, 0.1, device=device).unsqueeze(0) * gather_lengths
    ).type(torch.long) - 1

    binned_cumulative_dist = cumulative_dist.gather(1, gather_indices)
    binned_precisions = binned_cumulative_dist / (gather_indices + 1).type_as(
        binned_cumulative_dist
    )

    pl5 = binned_precisions[:, 1]
    pl2 = binned_precisions[:, 4]
    pl = binned_precisions[:, 9]
    auc = binned_precisions.mean(-1)

    return {"AUC": auc, "P@L": pl, "P@L2": pl2, "P@L5": pl5}

def evaluate_prediction(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> Dict[str, float]:
    if isinstance(targets, np.ndarray):
        targets = torch.from_numpy(targets)
    contact_ranges = [
        ("local", 3, 6),
        ("short", 6, 12),
        ("medium", 12, 24),
        ("long", 24, None),
    ]
    metrics = {}
    targets = targets.to(predictions.device)
    for name, minsep, maxsep in contact_ranges:
        rangemetrics = compute_precisions(
            predictions,
            targets,
            minsep=minsep,
            maxsep=maxsep,
        )
        for key, val in rangemetrics.items():
            metrics[f"{name}_{key}"] = val.item()
    return metrics
async def get_msacmap(seq,msa_model):
    try:
        msa_transformer=msa_model.model
        msa_transformer_alphabet = msa_model.alphabet
        msa_transformer_batch_converter = msa_transformer_alphabet.get_batch_converter()
        random_string = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(5))
        name=random_string
        data = [(name, seq)]
        msa_transformer_predictions = {}
        msa_transformer_results = []
        msa_transformer_batch_labels, msa_transformer_batch_strs, msa_transformer_batch_tokens=msa_transformer_batch_converter(data)
        msa_transformer_batch_tokens = msa_transformer_batch_tokens.to(next(msa_transformer.parameters()).device)
        msa_transformer_predictions[name] = msa_transformer.predict_contacts(msa_transformer_batch_tokens)[0].cpu()
        metrics = {"id": name, "model": "MSA Transformer (Unsupervised)"}
        predict_cmap=msa_transformer_predictions[name]
        print(predict_cmap.shape)
        npmascmap=predict_cmap.data.cpu().numpy()
        current_directory = os.getcwd()

        # Convert relative paths to absolute paths.
        absolute_path = os.path.join(current_directory, f"./out/msacmap/{name}.npy")

        print("msacmap path:", absolute_path)
        np.save(f"./out/msacmap/{name}.npy",npmascmap)
        if os.path.exists(absolute_path):
            return {"err_code": 0, "err_desc": "got msacmap, return msacmap path", "result": absolute_path}
        else:
            return {"err_code": -1, "err_desc": "failed to fetch msacmap, return msacmap path", "result": None}
    except RuntimeError as e:
        return {"err_code": -1, "err_desc": f"{e}", "result": None}
async def get_esmcmap(seq,msa_model):
    try:
        esm2=msa_model.model
        esm2_alphabet =msa_model.alphabet
        esm2_batch_converter = esm2_alphabet.get_batch_converter()
        esm2_predictions = {}
        esm2_results = []
        random_string = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(5))
        name=random_string
        data = [(name, seq)]
        esm2_batch_labels, esm2_batch_strs, esm2_batch_tokens = esm2_batch_converter(data)
        esm2_batch_tokens = esm2_batch_tokens.to(next(esm2.parameters()).device)
        esm2_predictions[name] = esm2.predict_contacts(esm2_batch_tokens)[0].cpu()
        metrics = {"id": name, "model": "MSA Transformer (Unsupervised)"}
        predict_cmap= esm2_predictions[name]
        print(predict_cmap)
        print(predict_cmap.shape)
    except RuntimeError as e:
        return {"err_code": -1, "err_desc": f"{e}", "result": None}
    



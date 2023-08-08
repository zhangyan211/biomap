# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Protein data type."""
import dataclasses
import io
import re
import collections

from typing import Any, Mapping, Optional, Sequence

import numpy as np

import residue_constants
from Bio.PDB import PDBParser
from Bio import Align
import pandas as pd
import os 

import json
import pandas as pd
import os
import numpy as np
import os
import pandas as pd
import glob
import io
from Bio.PDB import PDBParser, Select,PDBIO
restype_1to3 = {"A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN", "E": "GLU", "G": "GLY",
                "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO", "S": "SER",
                "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL","X":"UNK"}
restype_3to1 = {v: k for k, v in restype_1to3.items()}
IMGT = [
    range(1, 27),
    range(27, 39),
    range(39, 56),
    range(56, 66),
    range(66, 105),
    range(105, 118),
    range(118, 129),
]

def regions(seq, numbering, rng):
    fr1, fr2, fr3, fr4 = [], [], [], []
    cdr1, cdr2, cdr3 = [], [], []
    type_list = []

    for item in numbering[0][0][0]:
        (idx, key), aa = item
        sidx = "%d%s" % (idx, key.strip())  # str index
        if idx in rng[0]:  # fr1
            fr1.append([sidx, aa])
            type_list.append("fr1")
        elif idx in rng[1]:  # cdr1
            cdr1.append([sidx, aa])
            type_list.append("cdr1")
        elif idx in rng[2]:  # fr2
            fr2.append([sidx, aa])
            type_list.append("fr2")
        elif idx in rng[3]:  # cdr2
            cdr2.append([sidx, aa])
            type_list.append("cdr2")
        elif idx in rng[4]:  # fr3
            fr3.append([sidx, aa])
            type_list.append("fr3")
        elif idx in rng[5]:  # cdr3
            type_list.append("cdr3")
            cdr3.append([sidx, aa])
        elif idx in rng[6]:  # fr4
            fr4.append([sidx, aa])
            type_list.append("fr4")
        else:
            pass
            # logger.info(f"[WARNING] seq={seq}, sidx={sidx}, aa={aa}")

    return fr1, fr2, fr3, fr4, cdr1, cdr2, cdr3, type_list
from anarci import anarci
def make_numbering_by_api(
    seq, input_species=None, scheme="imgt", input_chain_type=None, ncpu=4
):
    seqs = [("0", seq)]
    numbering, alignment_details, hit_tables = anarci(
        seqs, scheme=scheme, output=False, ncpu=ncpu
    )
  

    species = alignment_details[0][0]["species"].lower()
    chain_type = alignment_details[0][0]["chain_type"].lower()
    e_value = alignment_details[0][0]["evalue"]
    score = alignment_details[0][0]["bitscore"]
    v_start = alignment_details[0][0]["query_start"]
    v_end = alignment_details[0][0]["query_end"]
    if scheme == "imgt":
        rng = IMGT
    elif scheme == "kabat" and chain_type.lower() == "h":
        rng = KABAT_H
    elif scheme == "kabat" and (chain_type.lower() == "l" or chain_type.lower() == "k"):
        rng = KABAT_L
    else:
        raise NotImplementedError
    fr1, fr2, fr3, fr4, cdr1, cdr2, cdr3, type_list = regions(seq, numbering, rng)
    str_fr1 = "".join([item[1] for item in fr1 if item[1] != "-"])
    str_fr2 = "".join([item[1] for item in fr2 if item[1] != "-"])
    str_fr3 = "".join([item[1] for item in fr3 if item[1] != "-"])
    str_fr4 = "".join([item[1] for item in fr4 if item[1] != "-"])
    str_cdr1 = "".join([item[1] for item in cdr1 if item[1] != "-"])
    str_cdr2 = "".join([item[1] for item in cdr2 if item[1] != "-"])
    str_cdr3 = "".join([item[1] for item in cdr3 if item[1] != "-"])
    str_overall = str_fr1 + str_cdr1 + str_fr2 + str_cdr2 + str_fr3 + str_cdr3 + str_fr4
    return str_overall

def is_antibody(seq, scheme='imgt', ncpu=4):
    seqs = [("0", seq)]
    numbering, alignment_details, hit_tables = anarci(seqs, scheme=scheme, output=False, ncpu=ncpu)
    if numbering[0] is None:
        return False, 'protein'
    chain_type = alignment_details[0][0]['chain_type'].lower()

    if chain_type is None:
        return False, 'protein'
    else:
        return True, chain_type

aligner = Align.PairwiseAligner()

FeatureDict = Mapping[str, np.ndarray]
ModelOutput = Mapping[str, Any]  # Is a nested dict.
PICO_TO_ANGSTROM = 0.01

# Complete sequence of chain IDs supported by the PDB format.
PDB_CHAIN_IDS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
PDB_MAX_CHAINS = len(PDB_CHAIN_IDS)  # := 62.


@dataclasses.dataclass(frozen=True)
class Protein:
    """Protein structure representation."""

    # Cartesian coordinates of atoms in angstroms. The atom types correspond to
    # residue_constants.atom_types, i.e. the first three are N, CA, CB.
    atom_positions: np.ndarray  # [num_res, num_atom_type, 3]

    # Amino-acid type for each residue represented as an integer between 0 and
    # 20, where 20 is 'X'.
    aatype: np.ndarray  # [num_res]

    # Binary float mask to indicate presence of a particular atom. 1.0 if an atom
    # is present and 0.0 if not. This should be used for loss masking.
    atom_mask: np.ndarray  # [num_res, num_atom_type]

    # Residue index as used in PDB. It is not necessarily continuous or 0-indexed.
    residue_index: np.ndarray  # [num_res]

    # B-factors, or temperature factors, of each residue (in sq. angstroms units),
    # representing the displacement of the residue from its ground truth mean
    # value.
    b_factors: np.ndarray  # [num_res, num_atom_type]

    # 0-indexed number corresponding to the chain in the protein that this
    # residue belongs to
    chain_index: np.ndarray

    # Optional remark about the protein. Included as a comment in output PDB
    # files
    remark: Optional[str] = None

    # Templates used to generate this protein (prediction-only)
    parents: Optional[Sequence[str]] = None

    # Chain corresponding to each parent
    parents_chain_index: Optional[Sequence[int]] = None

    # protein resolution
    resolution: any = None
    chain_ids: any = None

    def __post_init__(self):
        if len(np.unique(self.chain_index)) > PDB_MAX_CHAINS:
            raise ValueError(
                f"Cannot build an instance with more than {PDB_MAX_CHAINS} "
                "chains because these cannot be written to PDB format"
            )


def reorder_chain(protein_object):
    chain_ids = []
    [chain_ids.append(i) for i in protein_object.chain_index if not i in chain_ids]

    seq_to_entity_id = dict()
    aatype_group = collections.defaultdict(list)
    atom_positions_group = collections.defaultdict(list)
    atom_mask_group = collections.defaultdict(list)
    chain_index_group = collections.defaultdict(list)
    residue_index_group = collections.defaultdict(list)
    b_factors_group = collections.defaultdict(list)
    for chain_id in chain_ids:
        idx = protein_object.chain_index == chain_id
        aatype = protein_object.aatype[idx]
        seq = residue_constants.aatype_to_str_sequence(aatype)
        if seq not in seq_to_entity_id:
            seq_to_entity_id[seq] = len(seq_to_entity_id)
        aatype_group[seq_to_entity_id[seq]].append(aatype)
        atom_positions_group[seq_to_entity_id[seq]].append(
            protein_object.atom_positions[idx]
        )
        atom_mask_group[seq_to_entity_id[seq]].append(protein_object.atom_mask[idx])
        chain_index_group[seq_to_entity_id[seq]].append(protein_object.chain_index[idx])
        residue_index_group[seq_to_entity_id[seq]].append(
            protein_object.residue_index[idx]
        )
        b_factors_group[seq_to_entity_id[seq]].append(protein_object.b_factors[idx])
    aatype = []
    atom_positions = []
    chain_index = []
    residue_index = []
    b_factors = []
    atom_mask = []
    for id, _ in aatype_group.items():
        aatype.extend(aatype_group[id])
        atom_positions.extend(atom_positions_group[id])
        atom_mask.extend(atom_mask_group[id])
        residue_index.extend(residue_index_group[id])
        b_factors.extend(b_factors_group[id])
        chain_index.extend(chain_index_group[id])

    return Protein(
        atom_positions=np.concatenate(atom_positions),
        atom_mask=np.concatenate(atom_mask),
        aatype=np.concatenate(aatype),
        chain_index=np.concatenate(chain_index),
        residue_index=np.concatenate(residue_index),
        b_factors=np.concatenate(b_factors),
        resolution=protein_object.resolution,
    )


def filter_aa_from_seq(seq):
    aa_index = []
    for idx in range(len(seq)):
        if seq[idx] == "-":
            continue
        else:
            aa_index.append(idx)
    return aa_index


def from_pdb_string(
    pdb_str: str,
    chain_id: Optional[str] = None,
    use_filter_atom: bool = False,
    is_multimer: bool = False,
    return_id2seq: bool = False,
    is_cut_ab_fv: bool = False,
    resolution: float = None,
) -> Protein:
    """Takes a PDB string and constructs a Protein object.

    WARNING: All non-standard residue types will be converted into UNK. All
      non-standard atoms will be ignored.

    Args:
      pdb_str: The contents of the pdb file
      chain_id: If chain_id is specified (e.g. A), then only that chain is
      parsed. Else, all chains are parsed.

    Returns:
      A new `Protein` parsed from the pdb contents.
    """
    pdb_fh = io.StringIO(pdb_str)
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("none", pdb_fh)
    if resolution is None:
        resolution = structure.header["resolution"]
        if resolution is None:
            resolution = 0.0
    models = list(structure.get_models())
    model = models[0]
    pdb_chains = list(model.get_chains())
    chain_id_list = chain_id
    if chain_id is not None:
        if "," in chain_id:
            chain_id_list = chain_id.split(",")
        chain_model = []
        for chain_id in chain_id_list:
            for pdb_chain in pdb_chains:
                if chain_id == pdb_chain.id:
                    chain_model.append(pdb_chain)
        pdb_chains = chain_model

    atom_positions = []
    aatype = []
    atom_mask = []
    residue_index = []
    chain_ids = []
    b_factors = []
    id2seq = {}
    keep_aa_idx = []
    last_aa_idx = 0
    for chain in pdb_chains:
        cur_chain_aatype = []
        if chain_id is not None and chain.id not in chain_id_list:
            continue
        seqs = []
        for res in chain:
            if res.id[0] != " ":  # skip HETATM
                continue
            res_shortname = residue_constants.restype_3to1.get(res.resname, "X")
            seqs.append(res_shortname)
            restype_idx = residue_constants.restype_order.get(
                res_shortname, residue_constants.restype_num
            )
            pos = np.zeros((residue_constants.atom_type_num, 3))
            mask = np.zeros((residue_constants.atom_type_num,))
            res_b_factors = np.zeros((residue_constants.atom_type_num,))
            for atom in res:
                if atom.name not in residue_constants.atom_types:
                    continue
                pos[residue_constants.atom_order[atom.name]] = atom.coord
                mask[residue_constants.atom_order[atom.name]] = 1.0
                res_b_factors[residue_constants.atom_order[atom.name]] = atom.bfactor
            if np.sum(mask) < 0.5:
                # If no known atom positions are reported for the residue then skip it.
                continue

            aatype.append(restype_idx)
            cur_chain_aatype.append(restype_idx)
            atom_positions.append(pos)
            atom_mask.append(mask)
            residue_index.append(res.id[1])
            chain_ids.append(chain.id)
            b_factors.append(res_b_factors)
        if len(seqs)==0:
            continue
        seqs = "".join(seqs)
        id2seq[chain.id] = cur_chain_aatype
        if is_cut_ab_fv:
            flag_ab, _ = is_antibody(seqs, "imgt", 4)
            if flag_ab == True:
                numbering_seq = make_numbering_by_api(seqs)
                alignments = aligner.align(numbering_seq, seqs)
                alignment = next(alignments)
                alignment_seqs = str(alignment).strip().split("\n")
                aligned_n_seq = alignment_seqs[0].strip()
                index = filter_aa_from_seq(aligned_n_seq)
                id2seq[chain.id] = [cur_chain_aatype[idx] for idx in index]
                index = [i + last_aa_idx for i in index]

                keep_aa_idx.extend(index)
            else:
                index = [i + last_aa_idx for i in range(len(seqs))]
                keep_aa_idx.extend(index)
            last_aa_idx += len(seqs)
    if len(aatype)==0:
        raise ValueError
    if is_cut_ab_fv:
        aatype = [aatype[idx] for idx in keep_aa_idx]
        atom_positions = [atom_positions[idx] for idx in keep_aa_idx]
        atom_mask = [atom_mask[idx] for idx in keep_aa_idx]
        residue_index = [residue_index[idx] for idx in keep_aa_idx]
        chain_ids = [chain_ids[idx] for idx in keep_aa_idx]
        b_factors = [b_factors[idx] for idx in keep_aa_idx]
    # TODO: Parents logics from openfold:1.0
    # parents = None
    # parents_chain_index = None
    # if("PARENT" in pdb_str):
    #     parents = []
    #     parents_chain_index = []
    #     chain_id = 0
    #     for l in pdb_str.split("\n"):
    #         if("PARENT" in l):
    #             if(not "N/A" in l):
    #                 parent_names = l.split()[1:]
    #                 parents.extend(parent_names)
    #                 parents_chain_index.extend([
    #                     chain_id for _ in parent_names
    #                 ])
    #             chain_id += 1

    unique_chain_ids = np.unique(chain_ids)
    chain_id_mapping = {cid: n for n, cid in enumerate(unique_chain_ids)}
    chain_index = np.array([chain_id_mapping[cid] for cid in chain_ids])

    if return_id2seq:
        return (
            Protein(
                atom_positions=np.array(atom_positions),
                atom_mask=np.array(atom_mask),
                aatype=np.array(aatype),
                chain_index=chain_index,
                residue_index=np.array(residue_index),
                b_factors=np.array(b_factors),
                resolution=resolution,
                chain_ids=chain_ids,
            ),
            id2seq,
        )
    else:
        return Protein(
            atom_positions=np.array(atom_positions),
            atom_mask=np.array(atom_mask),
            aatype=np.array(aatype),
            chain_index=chain_index,
            residue_index=np.array(residue_index),
            b_factors=np.array(b_factors),
            resolution=resolution,
            chain_ids=chain_ids,
        )


def from_proteinnet_string(proteinnet_str: str) -> Protein:
    tag_re = r"(\[[A-Z]+\]\n)"
    tags = [tag.strip() for tag in re.split(tag_re, proteinnet_str) if len(tag) > 0]
    groups = zip(tags[0::2], [l.split("\n") for l in tags[1::2]])

    atoms = ["N", "CA", "C"]
    aatype = None
    atom_positions = None
    atom_mask = None
    for g in groups:
        if "[PRIMARY]" == g[0]:
            seq = g[1][0].strip()
            for i in range(len(seq)):
                if seq[i] not in residue_constants.restypes:
                    seq[i] = "X"
            aatype = np.array(
                [
                    residue_constants.restype_order.get(
                        res_symbol, residue_constants.restype_num
                    )
                    for res_symbol in seq
                ]
            )
        elif "[TERTIARY]" == g[0]:
            tertiary = []
            for axis in range(3):
                tertiary.append(list(map(float, g[1][axis].split())))
            tertiary_np = np.array(tertiary)
            atom_positions = np.zeros(
                (len(tertiary[0]) // 3, residue_constants.atom_type_num, 3)
            ).astype(np.float32)
            for i, atom in enumerate(atoms):
                atom_positions[:, residue_constants.atom_order[atom], :] = np.transpose(
                    tertiary_np[:, i::3]
                )
            atom_positions *= PICO_TO_ANGSTROM
        elif "[MASK]" == g[0]:
            mask = np.array(list(map({"-": 0, "+": 1}.get, g[1][0].strip())))
            atom_mask = np.zeros(
                (
                    len(mask),
                    residue_constants.atom_type_num,
                )
            ).astype(np.float32)
            for i, atom in enumerate(atoms):
                atom_mask[:, residue_constants.atom_order[atom]] = 1
            atom_mask *= mask[..., None]

    return Protein(
        atom_positions=atom_positions,
        atom_mask=atom_mask,
        aatype=aatype,
        residue_index=np.arange(len(aatype)),
        b_factors=None,
    )


def _chain_end(atom_index, end_resname, chain_name, residue_index) -> str:
    chain_end = "TER"
    return (
        f"{chain_end:<6}{atom_index:>5}      {end_resname:>3} "
        f"{chain_name:>1}{residue_index:>4}"
    )


def get_pdb_headers(prot: Protein, chain_id: int = 0) -> Sequence[str]:
    pdb_headers = []

    remark = prot.remark
    if remark is not None:
        pdb_headers.append(f"REMARK {remark}")

    parents = prot.parents
    parents_chain_index = prot.parents_chain_index
    if parents_chain_index is not None:
        parents = [p for i, p in zip(parents_chain_index, parents) if i == chain_id]

    if parents is None or len(parents) == 0:
        parents = ["N/A"]

    pdb_headers.append(f"PARENT {' '.join(parents)}")

    return pdb_headers


def add_pdb_headers(prot: Protein, pdb_str: str) -> str:
    """
    Add pdb headers to an existing PDB string. Useful during multi-chain
    recycling.
    """
    out_pdb_lines = []
    lines = pdb_str.split("\n")

    remark = prot.remark
    if remark is not None:
        out_pdb_lines.append(f"REMARK {remark}")

    parents_per_chain = None
    if prot.parents is not None and len(prot.parents) > 0:
        parents_per_chain = []
        if prot.parents_chain_index is not None:
            cur_chain = prot.parents_chain_index[0]
            parent_dict = {}
            for p, i in zip(prot.parents, prot.parents_chain_index):
                parent_dict.setdefault(str(i), [])
                parent_dict[str(i)].append(p)

            max_idx = max([int(chain_idx) for chain_idx in parent_dict])
            for i in range(max_idx + 1):
                chain_parents = parent_dict.get(str(i), ["N/A"])
                parents_per_chain.append(chain_parents)
        else:
            parents_per_chain.append(prot.parents)
    else:
        parents_per_chain = [["N/A"]]

    make_parent_line = lambda p: f"PARENT {' '.join(p)}"

    out_pdb_lines.append(make_parent_line(parents_per_chain[0]))

    chain_counter = 0
    for i, l in enumerate(lines):
        if "PARENT" not in l and "REMARK" not in l:
            out_pdb_lines.append(l)
        if "TER" in l and not "END" in lines[i + 1]:
            chain_counter += 1
            if not chain_counter >= len(parents_per_chain):
                chain_parents = parents_per_chain[chain_counter]
            else:
                chain_parents = ["N/A"]

            out_pdb_lines.append(make_parent_line(chain_parents))

    return "\n".join(out_pdb_lines)


def to_pdb(prot: Protein) -> str:
    """Converts a `Protein` instance to a PDB string.

    Args:
      prot: The protein to convert to PDB.

    Returns:
      PDB string.
    """
    restypes = residue_constants.restypes + ["X"]
    res_1to3 = lambda r: residue_constants.restype_1to3.get(restypes[r], "UNK")
    atom_types = residue_constants.atom_types

    pdb_lines = []

    # atom_mask = prot.atom_mask
    # aatype = prot.aatype
    # atom_positions = prot.atom_positions
    # residue_index = prot.residue_index.astype(np.int32)
    # chain_index = prot.chain_index.astype(np.int32)
    # b_factors = prot.b_factors
    
    ind = np.argsort(prot.residue_index)
    atom_mask = prot.atom_mask[ind]
    aatype = prot.aatype[ind]
    atom_positions = prot.atom_positions[ind]
    residue_index = prot.residue_index[ind].astype(np.int32)
    chain_index = prot.chain_index[ind].astype(np.int32)
    b_factors = prot.b_factors[ind]

    if np.any(aatype > residue_constants.restype_num):
        raise ValueError("Invalid aatypes.")

    # TODO: Headers building from openfold:1.0
    # headers = get_pdb_headers(prot)
    # if(len(headers) > 0):
    #     pdb_lines.extend(headers)

    # Construct a mapping from chain integer indices to chain ID strings.
    chain_ids = {}
    for i in np.unique(chain_index):  # np.unique gives sorted output.
        if i >= PDB_MAX_CHAINS:
            raise ValueError(
                f"The PDB format supports at most {PDB_MAX_CHAINS} chains."
            )
        chain_ids[i] = PDB_CHAIN_IDS[i]

    n = aatype.shape[0]
    pdb_lines.append("MODEL     1")
    atom_index = 1
    last_chain_index = chain_index[0]
    # Add all atom sites.
    for i in range(n):
        # Close the previous chain if in a multichain PDB.
        if last_chain_index != chain_index[i]:
            pdb_lines.append(
                _chain_end(
                    atom_index,
                    res_1to3(aatype[i - 1]),
                    chain_ids[chain_index[i - 1]],
                    residue_index[i - 1],
                )
            )
            last_chain_index = chain_index[i]
            atom_index += 1  # Atom index increases at the TER symbol.

        res_name_3 = res_1to3(aatype[i])
        for atom_name, pos, mask, b_factor in zip(
            atom_types, atom_positions[i], atom_mask[i], b_factors[i]
        ):
            if mask < 0.5:
                continue

            record_type = "ATOM"
            name = atom_name if len(atom_name) == 4 else f" {atom_name}"
            alt_loc = ""
            insertion_code = ""
            occupancy = 1.00
            element = atom_name[0]  # Protein supports only C, N, O, S, this works.
            charge = ""
            # PDB is a columnar format, every space matters here!
            atom_line = (
                f"{record_type:<6}{atom_index:>5} {name:<4}{alt_loc:>1}"
                f"{res_name_3:>3} {chain_ids[chain_index[i]]:>1}"
                f"{residue_index[i]:>4}{insertion_code:>1}   "
                f"{pos[0]:>8.3f}{pos[1]:>8.3f}{pos[2]:>8.3f}"
                f"{occupancy:>6.2f}{b_factor:>6.2f}          "
                f"{element:>2}{charge:>2}"
            )
            pdb_lines.append(atom_line)
            atom_index += 1

    # TODO: Finish update from Openfold:1.0
    # should_terminate = (i == n - 1)
    # if chain_index is not None:
    #     if(i != n - 1 and chain_index[i + 1] != prev_chain_index):
    #         should_terminate = True
    #         prev_chain_index = chain_index[i + 1]

    # if should_terminate:
    #     # Close the chain.
    #     chain_end = "TER"
    #     chain_termination_line = (
    #         f"{chain_end:<6}{atom_index:>5}      {res_1to3(aatype[-1]):>3} "
    #         f"{res_1to3(aatype[i]):>3} "
    #         f"{chain_tag:>1}{residue_index[i]:>4}"
    #     )
    #     pdb_lines.append(chain_termination_line)
    #     pdb_lines.append("ENDMDL")
    #     atom_index += 1

    #     if(i != n - 1):
    #         # "prev" is a misnomer here. This happens at the beginning of
    #         # each new chain.
    #         pdb_lines.extend(get_pdb_headers(prot, prev_chain_index))
    # Close the final chain.
    pdb_lines.append(
        _chain_end(
            atom_index,
            res_1to3(aatype[-1]),
            chain_ids[chain_index[-1]],
            residue_index[-1],
        )
    )

    pdb_lines.append("ENDMDL")
    pdb_lines.append("END")

    # Pad all lines to 80 characters
    pdb_lines = [line.ljust(80) for line in pdb_lines]
    return "\n".join(pdb_lines) + "\n"  # Add terminating newline.


def ideal_atom_mask(prot: Protein) -> np.ndarray:
    """Computes an ideal atom mask.

    `Protein.atom_mask` typically is defined according to the atoms that are
    reported in the PDB. This function computes a mask according to heavy atoms
    that should be present in the given sequence of amino acids.

    Args:
      prot: `Protein` whose fields are `numpy.ndarray` objects.

    Returns:
      An ideal atom mask.
    """
    return residue_constants.STANDARD_ATOM_MASK[prot.aatype]


def from_prediction(
    features: FeatureDict,
    result: ModelOutput,
    b_factors: Optional[np.ndarray] = None,
    chain_index: Optional[np.ndarray] = None,
    remark: Optional[str] = None,
    parents: Optional[Sequence[str]] = None,
    parents_chain_index: Optional[Sequence[int]] = None,
    remove_leading_feature_dimension: bool = False,
) -> Protein:
    """Assembles a protein from a prediction.

    Args:
      features: Dictionary holding model inputs. Need keys: aatype/asym_id, residue_index
      result: Dictionary holding model outputs. Need keys: (structure_module), final_atom_positions, final_atom_mask
      b_factors: (Optional) B-factors to use for the protein.
      chain_index: (Optional) Chain indices for multi-chain predictions
      remark: (Optional) Remark about the prediction
      parents: (Optional) List of template names
      remove_leading_feature_dimension: Whether to remove the leading dimension
        of the `features` values.

    Returns:
      A protein instance.
    """
    # To support Alphafold style output, need further unification
    fold_output = result["structure_module"] if "structure_module" in result else result
    if b_factors is None:
        b_factors = np.zeros_like(fold_output["final_atom_mask"])

    atom_mask = fold_output.get("final_atom_mask", result["final_atom_mask"])

    atom_positions = fold_output.get(
        "final_atom_positions", result["final_atom_positions"]
    )

    def _maybe_remove_leading_dim(arr: np.ndarray) -> np.ndarray:
        return arr[0] if remove_leading_feature_dimension else arr

    if "asym_id" in features:
        chain_index = _maybe_remove_leading_dim(features["asym_id"])
    else:
        chain_index = np.zeros_like(_maybe_remove_leading_dim(features["aatype"]))

    if b_factors is None:
        b_factors = np.zeros_like(result["final_atom_mask"])

    return Protein(
        aatype=_maybe_remove_leading_dim(features["aatype"]),
        atom_positions=atom_positions,
        atom_mask=atom_mask,
        residue_index=_maybe_remove_leading_dim(features["residue_index"]) + 1,
        b_factors=b_factors,
        chain_index=chain_index,
        remark=remark,
        parents=parents,
        parents_chain_index=parents_chain_index,
    )

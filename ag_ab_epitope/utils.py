from functools import cache
from pathlib import Path

import pandas as pd
from biopandas.pdb import PandasPdb


@cache
def load_pdb(pdb_path: Path | str) -> tuple[PandasPdb, dict[str, str]]:
    pdb = PandasPdb().read_pdb(str(pdb_path))
    seqs = pdb.amino3to1().groupby(["chain_id"])["residue_name"].sum().to_dict()

    return pdb, seqs


def find_epitope_by_distance(pdb_file: Path, vh: str, vl: str) -> pd.DataFrame:
    """Find the epitope by locating all heavy atoms within 4A of the antibody chains."""
    from pymol import cmd

    cmd.reinitialize()
    cmd.load(str(pdb_file), object="all_chains")
    ab_chain = f"(chain {vh} or chain {vl})"
    epitope_query = f"byres ( ((not {ab_chain}) within 4 of {ab_chain}) and (name C+CA+O+N) and polymer.protein)"

    epitope_space = {"indices": [], "chains": []}
    cmd.iterate(epitope_query, "indices.append(resi); chains.append(chain)", space=epitope_space)

    all_epitopes = pd.DataFrame(epitope_space)
    return (
        all_epitopes.assign(indices=lambda x: x["indices"].astype(int))
        .drop_duplicates()
        .sort_values(["chains", "indices"], ascending=True)
    )


def fix_pdb_chains(
    pdb_path: Path,
    ref_pdb_path: Path,
    output_pdb: Path | None,
    generated_chain_order: tuple[str, str, str],
    max_mismatch: int = 50,
) -> PandasPdb:
    """Assign chain names in reference PDB file to a given PDB."""
    pdb, _ = load_pdb(pdb_path)
    ref_pdb, _ = load_pdb(ref_pdb_path)
    col_names = pdb.df["ATOM"].columns

    # Get residues in both PDB files
    pdb_residues = (
        pdb.df["ATOM"]
        .filter(["residue_number", "residue_name"])
        .drop_duplicates()
        .reset_index(drop=True)
    )
    ref_pdb_residues = (
        ref_pdb.df["ATOM"]
        .filter(["chain_id", "residue_number", "residue_name"])
        .drop_duplicates()
        .assign(chain_id=lambda x: pd.Categorical(x["chain_id"], categories=generated_chain_order))
        .sort_values(["chain_id", "residue_number"])
        .rename(
            columns={"residue_name": "ref_residue_name", "residue_number": "ref_residue_number"}
        )
        .reset_index(drop=True)
        .assign(chain_id=lambda x: x["chain_id"].astype(str))
    )

    # Each chain should be the same in length, so we can concat them directly
    # assuming that chains are in the same order
    if pdb_residues.shape[0] != ref_pdb_residues.shape[0]:
        raise ValueError(
            f"Chains are not same in length ({pdb_residues.shape[0]} vs. {ref_pdb_residues.shape[0]}): {pdb_path} and {ref_pdb_path}"
        )
    fixed_pdb = pd.concat((pdb_residues, ref_pdb_residues), axis=1)
    if fixed_pdb.query("residue_name != ref_residue_name").shape[0] > max_mismatch:
        raise ValueError("Too many mismatching residues; check generated_chain_order")

    # Modify the original PDB and optionally save the output
    pdb.df["ATOM"] = (
        pdb.df["ATOM"]
        .drop(columns=["chain_id"])
        .merge(
            fixed_pdb.filter(["residue_number", "chain_id", "ref_residue_number"]),
            on="residue_number",
        )
        .drop(columns=["residue_number"])
        .rename(columns={"ref_residue_number": "residue_number"})
        .filter(col_names)
    )

    if output_pdb:
        pdb.to_pdb(str(output_pdb), records=["ATOM"])

    return pdb

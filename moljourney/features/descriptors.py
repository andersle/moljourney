"""Calculate molecular descriptors."""
from collections import defaultdict

import numpy as np
import pandas as pd
from mordred import Calculator as M_Calculator
from mordred import descriptors as m_descriptors
from rdkit.Chem import Descriptors, Descriptors3D, Mol
from rdkit.Contrib.IFG.ifg import identify_functional_groups
from rdkit.ML.Descriptors import MoleculeDescriptors
from tqdm.autonotebook import tqdm
from tqdm.contrib.concurrent import process_map


def calculate_mordred_descriptors(
    molecules: list[type[Mol]],
    ignore_3D: bool = False,
    nproc: int | None = None,
    quiet: bool = False,
    ipynb: bool = False,
) -> pd.DataFrame:
    """Calculate all descriptors from mordred"""
    calculator = M_Calculator(m_descriptors, ignore_3D=ignore_3D)
    return calculator.pandas(molecules, nproc=nproc, quiet=quiet, ipynb=ipynb)


def calculate_rdkit_descriptors(
    molecules: list[type[Mol]],
    descriptors: list[str] | None = None,
    disable_progress: bool = False,
    max_workers: int | None = None,
    chunksize: int = 1,
) -> pd.DataFrame:
    """Calculate RDKit 2D-descriptors for a set of molecules.

    Parameters
    ----------
    molecules : list of objects like rdkit.Chem.Mol
        The molecules to calculate descriptors for.
    descriptors : list of strings, optional
        The descriptors to calculate. If not given, this method will
        try all descriptors from rdkit.Chem.Descriptors._descList.
    disable_progress : boolean, optional
        If False, then we will display a progress bar.
    max_workers : integer, optional
        The maximum workers to use for concurrent.futures.ProcessPoolExecutor
        If None, use as many as processors.
    chunksize : int, optional
        Size of chunks for workers.
    """
    if descriptors is None:
        descriptors = [i[0] for i in Descriptors._descList]
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(
        descriptors
    )
    values = process_map(
        calculator.CalcDescriptors,
        molecules,
        max_workers=max_workers,
        disable=disable_progress,
        chunksize=chunksize,
    )
    return pd.DataFrame(np.array(values), columns=descriptors)


def calculate_rdkit_3d_descriptors(
    molecules: list[type[Mol]],
    disable_progress: bool = False,
) -> pd.DataFrame:
    """Calculate rdkit 3D-descriptors for a set of molecules.

    Note
    ----
    This method can not be run in parallel.
    There is only a small number of 3D descriptors in
    rdkit.Chem.Descriptors3D.
    """
    descriptors = [
        i for i in dir(Descriptors3D) if callable(getattr(Descriptors3D, i))
    ]
    functions = [getattr(Descriptors3D, i) for i in descriptors]
    # The functions above are all lambda functions -> they can not be pickled,
    # so we can not use process_map here...
    values = []
    for mol in tqdm(molecules, disable=disable_progress):
        values.append([func(mol) for func in functions])
    return pd.DataFrame(np.array(values), columns=descriptors)


def _fragments_rdkit() -> list[str]:
    """Get list of RDKit fragment descriptors."""
    return sorted(
        [i[0] for i in Descriptors._descList if i[0].startswith("fr_")]
    )


def calculate_rdkit_fragments(
    molecules: list[type[Mol]],
    disable_progress: bool = False,
    max_workers: int | None = None,
    chunksize: int = 1,
) -> pd.DataFrame:
    """Calculate RDKit fragment descriptors."""
    return calculate_rdkit_descriptors(
        molecules,
        descriptors=_fragments_rdkit(),
        disable_progress=disable_progress,
        max_workers=max_workers,
        chunksize=chunksize,
    )


def calculate_rdkit_functional_groups(
    molecules: list[type[Mol]],
    count: bool = False,
    disable_progress: bool = False,
) -> pd.DataFrame:
    """Calculate RDKit functional groups.

    Parameters
    ----------
    molecules : list of objects like rdkit.Chem.Mol
        The molecules to calculate descriptors for.
    count : bool, optional
        If True, we count the number of times each group appears
        in a molecule. If False, we do not count and only note
        if a group occurs (1) or not (0).
    disable_progress : boolean, optional
        If False, then we will display a progress bar.

    Note
    ----
    This method does not currently run in parallel due to some
    pickling issues with the identify_functional_groups method.
    """
    functional_groups = set([])
    all_groups = []

    for mol in tqdm(molecules, disable=disable_progress):
        groups = identify_functional_groups(mol)
        local = defaultdict(int)  # type: defaultdict[str, int]
        for group in groups:
            functional_groups.add(group.type)
            if count:
                local[group.type] += 1
            else:
                local[group.type] = 1
        all_groups.append(local)

    data = {
        key: [] for key in sorted(functional_groups)
    }  # type: dict[str, list[int]]
    for groups in all_groups:
        for key in data:
            data[key].append(groups.get(key, 0))
    return pd.DataFrame(data)

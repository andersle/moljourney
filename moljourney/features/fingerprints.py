"""Calculate fingerprint descriptors."""
import logging
from collections.abc import Callable
from functools import partial

import numpy as np
import pandas as pd
from rdkit.Avalon.pyAvalonTools import GetAvalonFP
from rdkit.Chem import MACCSkeys, Mol
from rdkit.Chem.AllChem import GetErGFingerprint
from rdkit.Chem.EState.Fingerprinter import FingerprintMol
from rdkit.Chem.rdFingerprintGenerator import (
    GetAtomPairGenerator,
    GetMorganGenerator,
    GetRDKitFPGenerator,
    GetTopologicalTorsionGenerator,
)
from tqdm.contrib.concurrent import process_map

LOGGER = logging.getLogger(__name__)

# Note: The methods below are mainly here so we can make use
# of process_map without too much hassle.


def maccs_as_numpy(mol: type[Mol]) -> np.ndarray:
    """Get MACCSKeys as a numpy array."""
    return np.array(
        [int(i) for i in MACCSkeys.GenMACCSKeys(mol).ToBitString()]
    )


def avalon_as_numpy(mol: type[Mol], bits: int = 2048) -> np.ndarray:
    """Get Avalon fingerprint as a numpy array."""
    return np.array(
        [int(i) for i in GetAvalonFP(mol, nBits=bits).ToBitString()]
    )


def estate_as_numpy(mol: type[Mol]) -> np.ndarray:
    """Get estate fingerprint as numpy array."""
    return FingerprintMol(mol)[0]


def erg_as_numpy(mol: type[Mol]) -> np.ndarray:
    """Get erg fingerprint as numpy array."""
    return GetErGFingerprint(mol)


def rdkit_as_numpy(mol: type[Mol], bits: int = 2048) -> np.ndarray:
    """Get RDKit fingerprint."""
    return GetRDKitFPGenerator(fpSize=bits).GetFingerprintAsNumPy(mol)


def rdkit_count_as_numpy(mol: type[Mol], bits: int = 2048) -> np.ndarray:
    """Get RDKit count fingerprint."""
    return GetRDKitFPGenerator(fpSize=bits).GetCountFingerprintAsNumPy(mol)


def morgan_as_numpy(
    mol: type[Mol], bits: int = 2048, radius: int = 2
) -> np.ndarray:
    """Get Morgan fingerprint with radius 2."""
    return GetMorganGenerator(
        fpSize=bits, radius=radius
    ).GetFingerprintAsNumPy(mol)


def morgan_count_as_numpy(
    mol: type[Mol], bits: int = 2048, radius: int = 2
) -> np.ndarray:
    """Get Morgan count fingerprint with radius 2."""
    return GetMorganGenerator(
        fpSize=bits, radius=radius
    ).GetCountFingerprintAsNumPy(mol)


def atompair_as_numpy(mol: type[Mol], bits: int = 2048) -> np.ndarray:
    """Get atom pair fingerprint."""
    return GetAtomPairGenerator(fpSize=bits).GetFingerprintAsNumPy(mol)


def atompair_count_as_numpy(mol: type[Mol], bits: int = 2048) -> np.ndarray:
    """Get atom pair count fingerprint."""
    return GetAtomPairGenerator(fpSize=bits).GetCountFingerprintAsNumPy(mol)


def tt_as_numpy(mol: type[Mol], bits: int = 2048) -> np.ndarray:
    """Get topological torsion fingerprint."""
    return GetTopologicalTorsionGenerator(fpSize=bits).GetFingerprintAsNumPy(
        mol
    )


def tt_count_as_numpy(mol: type[Mol], bits: int = 2048) -> np.ndarray:
    """Get topological torsion count fingerprint."""
    return GetTopologicalTorsionGenerator(
        fpSize=bits
    ).GetCountFingerprintAsNumPy(mol)


def get_fingerprints(
    molecules: list[type[Mol]],
    method: Callable[[type[Mol]], np.ndarray] | str = "maccs",
    bits: int = 2048,
    disable_progress: bool = False,
    max_workers: int | None = None,
    chunksize: int = 1,
):
    """Calculate fingerprints for molecules.

    Parameters
    ----------
    molecules : list of objects like rdkit.Chem.Mol
        The molecules to calculate descriptors for.
    method : string or callable
        The method used to calculate the fingerprint.
    bits : integer, optional
        Number of bits to use for the calculation of the fingerprint.
        This is ignored if method is a callable.
        disable_progress : boolean, optional
        If False, then we will display a progress bar.
    max_workers : integer, optional
        The maximum workers to use for concurrent.futures.ProcessPoolExecutor
        If None, use as many as processors.
    chunksize : int, optional
        Size of chunks for workers.

    Returns
    -------
    out : object like pd.DataFrame
        The calculated bits (columns) for the molecules (rows).
    """
    methods = {
        "rdkit": partial(rdkit_as_numpy, bits=bits),
        "rdkit-count": partial(rdkit_count_as_numpy, bits=bits),
        "morgan2": partial(morgan_as_numpy, bits=bits, radius=2),
        "morgan2-count": partial(morgan_count_as_numpy, bits=bits, radius=2),
        "morgan3": partial(morgan_as_numpy, bits=bits, radius=3),
        "morgan3-count": partial(morgan_count_as_numpy, bits=bits, radius=3),
        "topologicaltorsion": partial(tt_as_numpy, bits=bits),
        "topologicaltorsion-count": partial(tt_count_as_numpy, bits=bits),
        "atompair": partial(atompair_as_numpy, bits=bits),
        "atompair-count": partial(atompair_count_as_numpy, bits=bits),
        "avalon": partial(avalon_as_numpy, bits=bits),
        "erg": erg_as_numpy,
        "maccs": maccs_as_numpy,
        "estate": estate_as_numpy,
    }

    LOGGER.debug('Using method "%s" for fingerprint.', method)
    if not callable(method):
        try:
            calculator = methods[method]
        except KeyError:
            LOGGER.error(
                'Unknown fingerprint method "%s". Use one of: %s',
                method,
                sorted(list(methods.keys())),
            )
            return None
    else:
        calculator = method

    values = process_map(
        calculator,
        molecules,
        max_workers=max_workers,
        disable=disable_progress,
        chunksize=chunksize,
    )
    matrix = np.array(values, dtype=int)
    bit = [f"bit_{i+1}" for i in range(matrix.shape[1])]
    return pd.DataFrame(matrix, columns=bit)

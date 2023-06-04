import numpy as np
from rdkit.Chem import MACCSkeys

from moljourney.features.fingerprints import (
    atompair_as_numpy,
    atompair_count_as_numpy,
    avalon_as_numpy,
    erg_as_numpy,
    estate_as_numpy,
    get_fingerprints,
    get_fingerprints_selection,
    maccs_as_numpy,
    morgan_as_numpy,
    morgan_count_as_numpy,
    rdkit_as_numpy,
    rdkit_count_as_numpy,
    tt_as_numpy,
    tt_count_as_numpy,
)
from moljourney.molecules import molecules_from_smiles

SMILES = ["C", "CC", "CCC", "CO", "CC(O)=O", "OCCCO"]
MOLECULES, _ = molecules_from_smiles(SMILES)


def my_method(mol):
    return np.array(
        [int(i) for i in MACCSkeys.GenMACCSKeys(mol).ToBitString()]
    )


def test_get_fingerprints():
    """Test calling the get_fingerprint method."""
    # Test one that does not exist:
    fp0 = get_fingerprints(
        MOLECULES,
        method="should-not-exist",
        bits=256,
        disable_progress=True,
        max_workers=1,
    )
    assert fp0 is None
    # Test the MACCSkeys ones:
    fp1 = get_fingerprints(
        MOLECULES,
        method="maccs",
        bits=256,
        disable_progress=True,
        max_workers=1,
    )
    assert fp1 is not None
    assert fp1.shape == (6, 167)
    # Test a custom one:
    fp2 = get_fingerprints(
        MOLECULES,
        method=my_method,
        bits=256,
        disable_progress=True,
        max_workers=1,
    )
    assert fp2 is not None
    assert fp2.shape == (6, 167)
    assert np.allclose(fp1.to_numpy(), fp2.to_numpy())


def test_all_fp_methods():
    methods = [
        maccs_as_numpy,
        avalon_as_numpy,
        estate_as_numpy,
        erg_as_numpy,
        rdkit_as_numpy,
        rdkit_count_as_numpy,
        morgan_as_numpy,
        morgan_count_as_numpy,
        atompair_as_numpy,
        atompair_count_as_numpy,
        tt_as_numpy,
        tt_count_as_numpy,
    ]
    for method in methods:
        fpi = method(MOLECULES[-1])
        assert isinstance(fpi, np.ndarray)


def test_fingerprint_selection(monkeypatch):
    """Test that we can calculate the selection."""
    monkeypatch.setattr(
        "moljourney.features.fingerprints.METHODS_BITS", ["rdkit"]
    )
    monkeypatch.setattr(
        "moljourney.features.fingerprints.METHODS_NO_BITS", ["maccs"]
    )
    data = get_fingerprints_selection(
        MOLECULES,
        bits=[8, 32],
    )
    assert "rdkit-8" in data
    assert "rdkit-32" in data
    assert "maccs" in data
    data = get_fingerprints_selection(
        MOLECULES,
    )
    assert "rdkit-512" in data
    assert "rdkit-1024" in data
    assert "rdkit-2048" in data
    assert "rdkit-4096" in data
    assert "maccs" in data

from moljourney.features.descriptors import (
    calculate_mordred_descriptors,
    calculate_rdkit_2d_descriptors,
)
from moljourney.molecules import molecules_from_smiles

SMILES = ["C", "CC", "CCC", "CCCC"]


def test_use_mordred():
    """Test that we can use mordred."""
    molecules, _ = molecules_from_smiles(SMILES)
    mordred = calculate_mordred_descriptors(
        molecules,
        ignore_3D=False,
        nproc=1,
        quiet=True,
        ipynb=False,
    )
    assert mordred.shape == (4, 1826)
    assert "ABC" in mordred


def test_rdkit_2d():
    """Test that we can use RDKit for descriptors."""
    molecules, _ = molecules_from_smiles(SMILES)
    rdk = calculate_rdkit_2d_descriptors(
        molecules,
        descriptors=None,
        disable_progress=True,
        max_workers=1,
        chunksize=1,
        leave=True,
    )
    assert rdk.shape == (4, 208)
    assert "ExactMolWt" in rdk
    # Test selection of descriptors:
    rdk = calculate_rdkit_2d_descriptors(
        molecules,
        descriptors=["MaxEStateIndex", "MinEStateIndex"],
        disable_progress=True,
        max_workers=1,
        chunksize=1,
        leave=True,
    )
    assert rdk.shape == (4, 2)
    assert "MaxEStateIndex" in rdk
    assert "MinEStateIndex" in rdk
    # Test a descriptor that does not exists
    rdk = calculate_rdkit_2d_descriptors(
        molecules,
        descriptors=[
            "WellWellWell",
        ],
        disable_progress=True,
        max_workers=1,
        chunksize=1,
        leave=True,
    )
    assert rdk.shape == (4, 1)
    assert "WellWellWell" in rdk
    assert all([i == 777 for i in rdk["WellWellWell"].values])

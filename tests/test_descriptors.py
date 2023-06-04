from moljourney.features.descriptors import (
    calculate_mordred_descriptors,
    calculate_rdkit_2d_descriptors,
    calculate_rdkit_3d_descriptors,
    calculate_rdkit_fragments,
    calculate_rdkit_functional_groups,
    calculate_rdkit_descriptors,
)
from moljourney.molecules import molecules_from_smiles

SMILES = ["C", "CC", "CCC", "CO", "CC(O)=O", "OCCCO"]
MOLECULES, _ = molecules_from_smiles(SMILES)


def test_use_mordred():
    """Test that we can use mordred."""
    mordred = calculate_mordred_descriptors(
        MOLECULES,
        ignore_3D=False,
        nproc=1,
        quiet=True,
        ipynb=False,
    )
    assert mordred.shape == (6, 1826)
    assert "ABC" in mordred


def test_rdkit_2d():
    """Test that we can use RDKit for descriptors."""
    rdk = calculate_rdkit_2d_descriptors(
        MOLECULES,
        descriptors=None,
        disable_progress=True,
        max_workers=1,
        chunksize=1,
        leave=True,
    )
    assert rdk.shape == (6, 208)
    assert "ExactMolWt" in rdk
    # Test selection of descriptors:
    rdk = calculate_rdkit_2d_descriptors(
        MOLECULES,
        descriptors=["MaxEStateIndex", "MinEStateIndex"],
        disable_progress=True,
        max_workers=1,
        chunksize=1,
        leave=True,
    )
    assert rdk.shape == (6, 2)
    assert "MaxEStateIndex" in rdk
    assert "MinEStateIndex" in rdk
    # Test a descriptor that does not exists
    rdk = calculate_rdkit_2d_descriptors(
        MOLECULES,
        descriptors=[
            "WellWellWell",
        ],
        disable_progress=True,
        max_workers=1,
        chunksize=1,
        leave=True,
    )
    assert rdk.shape == (6, 1)
    assert "WellWellWell" in rdk
    assert all([i == 777 for i in rdk["WellWellWell"].values])


def test_rdkit_3d():
    """Test that we can use the RKDit 3D descriptors."""
    rdk = calculate_rdkit_3d_descriptors(MOLECULES, disable_progress=True)
    assert rdk.shape == (6, 10)
    assert "Asphericity" in rdk


def test_rdkit_fragments():
    """Test the fragment calculation."""
    rdk = calculate_rdkit_fragments(MOLECULES, disable_progress=True, max_workers=1)
    assert rdk.shape == (6, 85)
    assert all(i.startswith("fr_") for i in rdk.columns)


def test_rdkit_ifg():
    """Test the functional group identification."""
    rdk = calculate_rdkit_functional_groups(MOLECULES, count=False, disable_progress=True)
    assert rdk.shape == (6, 2)
    assert "CC(=O)O" in rdk
    assert "CO" in rdk
    assert max(rdk["CO"]) == 1
    rdk = calculate_rdkit_functional_groups(MOLECULES, count=True, disable_progress=True)
    assert max(rdk["CO"]) == 2


def test_rdkit_descriptors_all():
    """Test that we can calculate all RDKit descriptors."""
    rdk = calculate_rdkit_descriptors(MOLECULES, disable_progress=True, max_workers=1, count=False)
    assert max(rdk["CO"]) == 1
    assert rdk.shape == (6, 220)
    rdk = calculate_rdkit_descriptors(MOLECULES, disable_progress=True, max_workers=1, count=True)
    assert max(rdk["CO"]) == 2
    assert rdk.shape == (6, 220)

from requests.models import CONTENT_CHUNK_SIZE
from moljourney.molecules import molecule_from_smile, molecules_from_smiles
from rdkit.Chem import Mol



def test_molecule_from_smile():
    """Test that we can generate molecules from smiles."""
    mol, svg = molecule_from_smile("CCC", svg=False)
    assert svg is None
    assert isinstance(mol, Mol)
    mol, svg = molecule_from_smile("CCC", svg=True)
    assert isinstance(mol, Mol)
    assert isinstance(svg, str)


def test_molecules_from_smiles():
    smiles = ["C", "CC", "CCC", "CCCC"]
    molecules, svgs = molecules_from_smiles(
        smiles, svg=True, disable_progress=True, max_workers=2, chunksize=1,
    )
    assert len(molecules) == 4
    assert len(svgs) == 4


if __name__ == "__main__":
    test_molecule_from_smile()

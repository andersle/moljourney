"""Methods for dealing with molecules."""
from rdkit import Chem
from rdkit.Chem import (
    Draw,
    Mol,
    rdCoordGen,
)
from tqdm.autonotebook import tqdm


def make_svg(mol: type[Mol], sizex: int = 200, sizey: int = 200) -> str:
    """Make SVG text for a given rdkit molecule."""
    drawing = Draw.rdMolDraw2D.MolDraw2DSVG(sizex, sizey)
    drawing.DrawMolecule(mol)
    drawing.FinishDrawing()
    return drawing.GetDrawingText()


def make_molecules_from_smiles(
    smiles: list[str],
    svg: bool = False,
    sizex: int = 200,
    sizey: int = 200,
    disable_progress: bool = True,
) -> tuple[list[type[Mol]], list[str]]:
    """Generate rdkit molecules from smiles.

    Parameters
    ----------
    smiles :  list of strings
        The smiles to process.
    svg : boolean, optional
        If True, this method will also generate SVG images
        for the molecules.
    sizex : int, optional
        Width (pixels) of the generated SVG image.
    sizey : int, optional
        Height (pixels) of the generated SVG image.
    disable_progress : boolean, optional
        If False, then we will display a progress bar.

    Returns
    -------
    molecules : list of objects like rdkit.Chem.Mol
        The generated molecules.
    svgs : list of strings
        The generated SVG images.
    """
    molecules = []
    svgs = []
    for smilei in tqdm(smiles, disable=disable_progress):
        mol = Chem.MolFromSmiles(smilei)
        rdCoordGen.AddCoords(mol)
        molecules.append(mol)
        if svg:
            svgs.append(make_svg(mol, sizex=sizex, sizey=sizey))
    return molecules, svgs

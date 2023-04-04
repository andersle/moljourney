"""Methods for dealing with molecules."""
from functools import partial

from rdkit import Chem
from rdkit.Chem import (
    Draw,
    Mol,
    rdCoordGen,
)
from tqdm.contrib.concurrent import process_map


def make_svg(mol: type[Mol], sizex: int = 200, sizey: int = 200) -> str:
    """Make SVG text for a given rdkit molecule."""
    drawing = Draw.rdMolDraw2D.MolDraw2DSVG(sizex, sizey)
    drawing.DrawMolecule(mol)
    drawing.FinishDrawing()
    return drawing.GetDrawingText()


def create_molecule(
    smile: str, svg: bool = False, sizex: int = 200, sizey: int = 200
) -> tuple[type[Mol], str | None]:
    mol = Chem.MolFromSmiles(smile)
    rdCoordGen.AddCoords(mol)
    if svg:
        return mol, make_svg(mol, sizex=sizex, sizey=sizey)
    return mol, None


def molecules_from_smiles(
    smiles: list[str],
    svg: bool = False,
    sizex: int = 200,
    sizey: int = 200,
    disable_progress: bool = False,
    max_workers: int | None = None,
    chunksize: int = 1,
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
    max_workers : integer, optional
        The maximum workers to use for concurrent.futures.ProcessPoolExecutor
        If None, use as many as processors.
    chunksize : int, optional
        Size of chunks for workers.


    Returns
    -------
    molecules : list of objects like rdkit.Chem.Mol
        The generated molecules.
    svgs : list of strings
        The generated SVG images.
    """
    method = partial(
        create_molecule,
        svg=svg,
        sizex=sizex,
        sizey=sizey,
    )

    values = process_map(
        method,
        smiles,
        max_workers=max_workers,
        disable=disable_progress,
        chunksize=chunksize,
    )
    molecules = [i[0] for i in values]
    svgs = [i[1] for i in values] if svg else []
    return molecules, svgs

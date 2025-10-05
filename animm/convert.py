"""Lightweight conversions -> ASE ``Atoms``.

Primary function: ``smiles_to_ase`` using RDKit when present, else a limited
fallback via ``ase.build.molecule``.
"""

from __future__ import annotations

from ase import Atoms
from ase.build import molecule

try:
    from rdkit import Chem  # type: ignore
    from rdkit.Chem import AllChem  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    Chem = None  # type: ignore
    AllChem = None  # type: ignore


def smiles_to_ase(smiles: str, add_h: bool = True, conformer_id: int = 0) -> Atoms:
    """Convert a SMILES string to an ASE Atoms object via RDKit (if installed) or fallback.

    If RDKit is not installed, tries ase.build.molecule as a simple fallback (works only for
    a limited set of common molecules / formulas).
    """
    if Chem is None:
        # Fallback: interpret as a known molecule name or formula
        return molecule(smiles)

    mol = Chem.MolFromSmiles(smiles)  # type: ignore[attr-defined]
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    if add_h:
        mol = Chem.AddHs(mol)  # type: ignore[attr-defined]
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())  # type: ignore[attr-defined]
    AllChem.UFFOptimizeMolecule(mol)  # type: ignore[attr-defined]
    if conformer_id >= mol.GetNumConformers():
        raise ValueError("Conformer index out of range")

    conf = mol.GetConformer(conformer_id)
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    positions = []
    for i in range(conf.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        positions.append((pos.x, pos.y, pos.z))

    return Atoms(symbols=symbols, positions=positions)

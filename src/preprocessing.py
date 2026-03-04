import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors

# -------------------------------------------------------
# 所有 RDKit 標準 descriptor 的名稱清單
# -------------------------------------------------------
# ALL_RDKIT_DESCRIPTOR_NAMES = [name for name, _ in Descriptors.descList]

ALL_RDKIT_DESCRIPTOR_NAMES = [desc_name[0] for desc_name in Descriptors._descList]

# 用完整的 RDKit descriptor 清單初始化 calculator
_calculator = MoleculeDescriptors.MolecularDescriptorCalculator(ALL_RDKIT_DESCRIPTOR_NAMES)


def clip_array(X, clip_value=1e6):
    return np.clip(X, -clip_value, clip_value)


# -------------------------------------------------------
# Polymer-aware features
# -------------------------------------------------------

def atom_fraction_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
    total = mol.GetNumAtoms()

    def frac(x):
        return atoms.count(x) / total if total > 0 else 0

    return {
        "frac_C": frac("C"),
        "frac_O": frac("O"),
        "frac_N": frac("N"),
        "frac_S": frac("S"),
        "frac_hetero": 1 - frac("C"),
    }


def rigidity_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    aromatic_atoms = sum(a.GetIsAromatic() for a in mol.GetAtoms())

    return {
        "aromatic_ring_count": rdMolDescriptors.CalcNumAromaticRings(mol),
        "aromatic_atom_frac": aromatic_atoms / mol.GetNumAtoms(),
        "double_bond_count": sum(
            b.GetBondTypeAsDouble() == 2 for b in mol.GetBonds()
        ),
    }


def flexibility_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    num_bonds = mol.GetNumBonds()

    return {
        "rotatable_bond_frac": (
            rdMolDescriptors.CalcNumRotatableBonds(mol) / num_bonds
            if num_bonds > 0
            else 0
        ),
        "heavy_atom_count": mol.GetNumHeavyAtoms(),
    }


def polymer_aware_features(smiles):
    """計算所有 polymer-aware features，回傳 dict。"""
    features = {}
    for func in [atom_fraction_features, rigidity_features, flexibility_features]:
        out = func(smiles)
        if out is not None:
            features.update(out)
    return features

def clip_array(X, clip_value=1e6):
    return np.clip(X, -clip_value, clip_value)

# -------------------------------------------------------
# RDKit descriptor 計算
# -------------------------------------------------------

def calc_rdkit_descriptors(smiles: str) -> dict:
    """計算所有 RDKit 標準 descriptors，回傳 {name: value} dict。"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {name: None for name in ALL_RDKIT_DESCRIPTOR_NAMES}
    values = _calculator.CalcDescriptors(mol)
    return dict(zip(ALL_RDKIT_DESCRIPTOR_NAMES, values))
import sys
import os
from functools import lru_cache

PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PARENT_DIR)

from rdkit import Chem
from rdkit.Chem import Descriptors

from nova_ph2.utils import get_heavy_atom_count
from nova_ph2.combinatorial_db.reactions import get_smiles_from_reaction

def validate_molecules_sampler(
    sampler_data: dict,
    config: dict,
) -> tuple[list[str], list[str]]:
    
    molecules = sampler_data["molecules"]

    valid_smiles = []
    valid_names = []
                
    for molecule in molecules:
        try:
            if molecule is None:
                continue
            
            smiles = get_smiles_from_reaction_cached(molecule)
            if not smiles:
                continue
            
            if get_heavy_atom_count(smiles) < config['min_heavy_atoms']:
                continue

            try:    
                num_rotatable_bonds = rotatable_bonds_cached(smiles)
                if num_rotatable_bonds < config['min_rotatable_bonds'] or num_rotatable_bonds > config['max_rotatable_bonds'] or num_rotatable_bonds is None:
                    continue
            except Exception as e:
                continue
    
            valid_smiles.append(smiles)
            valid_names.append(molecule)
        except Exception as e:
            continue
        
    return valid_names, valid_smiles

@lru_cache(maxsize=200_000)
def get_smiles_from_reaction_cached(name: str) -> str | None:
    try:
        return get_smiles_from_reaction(name)
    except Exception:
        return None

@lru_cache(maxsize=200_000)
def mol_from_smiles_cached(s: str):
    try:
        return Chem.MolFromSmiles(s)
    except Exception:
        return None

@lru_cache(maxsize=200_000)
def rotatable_bonds_cached(s: str) -> int | None:
    mol = mol_from_smiles_cached(s)
    if mol is None:
        return None
    try:
        return Descriptors.NumRotatableBonds(mol)
    except Exception:
        return None
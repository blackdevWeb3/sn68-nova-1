import sqlite3
import random
import os
import json
from typing import List, Tuple, Optional
import bittensor as bt
from rdkit import Chem
from tqdm import tqdm
from functools import lru_cache

from miner_utils import validate_molecules_sampler
from nova_ph2.combinatorial_db.reactions import (
    get_reaction_info, 
    get_smiles_from_reaction
)

@lru_cache(maxsize=8)
def get_available_reactions(db_path: str = None) -> List[Tuple[int, str, int, int, int]]:
    """
    Get all available reactions from the database.
    
    Args:
        db_path: Path to the molecules database
        
    Returns:
        List of tuples (rxn_id, smarts, roleA, roleB, roleC)
    """
    if db_path is None:
        db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "combinatorial_db", "molecules.sqlite"))
    
    try:
        abs_db_path = os.path.abspath(db_path)
        conn = sqlite3.connect(f"file:{abs_db_path}?mode=ro&immutable=1", uri=True)
        conn.execute("PRAGMA query_only = ON")
        cursor = conn.cursor()
        cursor.execute("SELECT rxn_id, smarts, roleA, roleB, roleC FROM reactions")
        results = cursor.fetchall()
        conn.close()
        return results
    except Exception as e:
        bt.logging.error(f"Error getting available reactions: {e}")
        return []


@lru_cache(maxsize=1024)
def get_molecules_by_role(role_mask: int, db_path: str) -> List[Tuple[int, str, int]]:
    """
    Get all molecules that have the specified role_mask.
    
    Args:
        role_mask: The role mask to filter by
        db_path: Path to the molecules database
        
    Returns:
        List of tuples (mol_id, smiles, role_mask) for molecules that match the role
    """
    try:
        abs_db_path = os.path.abspath(db_path)
        conn = sqlite3.connect(f"file:{abs_db_path}?mode=ro&immutable=1", uri=True)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT mol_id, smiles, role_mask FROM molecules WHERE (role_mask & ?) = ?", 
            (role_mask, role_mask)
        )
        results = cursor.fetchall()
        conn.close()
        return results
    except Exception as e:
        bt.logging.error(f"Error getting molecules by role {role_mask}: {e}")
        return []

def generate_valid_random_molecules_batch(rxn_id: int, n_samples: int, db_path: str, subnet_config: dict, 
                                 batch_size: int = 200, seed: int = None,
                                 elite_names: list[str] = None, elite_frac: float = 0.5, mutation_prob: float = 0.1,
                                 avoid_inchikeys: set[str] = None) -> dict:
    """
    Efficiently generate n_samples valid molecules by generating them in batches and validating.
    """
    reaction_info = get_reaction_info(rxn_id, db_path)
    if not reaction_info:
        bt.logging.error(f"Could not get reaction info for rxn_id {rxn_id}")
        return {"molecules": [None] * n_samples}
    
    smarts, roleA, roleB, roleC = reaction_info
    is_three_component = roleC is not None and roleC != 0
    
    molecules_A = get_molecules_by_role(roleA, db_path)
    molecules_B = get_molecules_by_role(roleB, db_path)
    molecules_C = get_molecules_by_role(roleC, db_path) if is_three_component else []
    
    if not molecules_A or not molecules_B or (is_three_component and not molecules_C):
        bt.logging.error(f"No molecules found for roles A={roleA}, B={roleB}, C={roleC}")
        return {"molecules": [None] * n_samples}

    valid_molecules = []
    valid_smiles = []
    seen_keys = set()
    iteration = 0

    progress_bar = tqdm(total=n_samples, desc="Creating valid molecules", unit="molecule", miniters=100, mininterval=0.5)
    
    while len(valid_molecules) < n_samples:
        iteration += 1
        
        needed = n_samples - len(valid_molecules)
        
        batch_size_actual = min(batch_size, needed * 2)
        
        emitted_names = set()
        if elite_names:
            n_elite = max(0, min(batch_size_actual, int(batch_size_actual * elite_frac)))
            n_rand = batch_size_actual - n_elite

            elite_batch = generate_offspring_from_elites(
                rxn_id=rxn_id,
                n=n_elite,
                elite_names=elite_names,
                molecules_A=molecules_A,
                molecules_B=molecules_B,
                molecules_C=molecules_C,
                is_three_component=is_three_component,
                mutation_prob=mutation_prob,
                seed=seed,
                avoid_names=emitted_names,
                avoid_inchikeys=avoid_inchikeys,
                max_tries=10
            )
            emitted_names.update(elite_batch)

            rand_batch = generate_molecules_from_pools(
                rxn_id, n_rand, molecules_A, molecules_B, molecules_C, is_three_component, seed
            )
            rand_batch = [n for n in rand_batch if n and (n not in emitted_names)]
            batch_molecules = elite_batch + rand_batch
        else:
            batch_molecules = generate_molecules_from_pools(
                rxn_id, batch_size_actual, molecules_A, molecules_B, molecules_C, is_three_component, seed
            )
        
        batch_sampler_data = {"molecules": batch_molecules}
        batch_valid_molecules, batch_valid_smiles = validate_molecules_sampler(batch_sampler_data, subnet_config)

        added = 0
        for i, name in enumerate(batch_valid_molecules):
            if not name:
                continue
            s = batch_valid_smiles[i] if i < len(batch_valid_smiles) else None
            if not s:
                continue
            try:
                key = smiles_to_inchikey(s)
                if not key:
                    continue
            except Exception:
                continue
            if key in seen_keys or (avoid_inchikeys and key in avoid_inchikeys):
                continue

            seen_keys.add(key)
            valid_molecules.append(name)
            valid_smiles.append(s)
            added += 1
        
        progress_bar.update(added)
    
    final_molecules = valid_molecules[:n_samples]
    final_smiles = valid_smiles[:n_samples]
    progress_bar.close()

    return {"molecules": final_molecules,"smiles": final_smiles}


def generate_molecules_from_pools(rxn_id: int, n: int, molecules_A: List[Tuple], molecules_B: List[Tuple], 
                               molecules_C: List[Tuple], is_three_component: bool, seed: int = None) -> List[str]:
    rng = random.Random(seed) if seed is not None else random

    A_ids = [a[0] for a in molecules_A]
    B_ids = [b[0] for b in molecules_B]
    C_ids = [c[0] for c in molecules_C] if is_three_component else None

    picks_A = rng.choices(A_ids, k=n)
    picks_B = rng.choices(B_ids, k=n)
    if is_three_component:
        picks_C = rng.choices(C_ids, k=n)
        names = [f"rxn:{rxn_id}:{a}:{b}:{c}" for a, b, c in zip(picks_A, picks_B, picks_C)]
    else:
        names = [f"rxn:{rxn_id}:{a}:{b}" for a, b in zip(picks_A, picks_B)]

    names = list(dict.fromkeys(names))
    return names

def _parse_components(name: str) -> tuple[int, int, int | None]:
    # name format: "rxn:{rxn_id}:{A}:{B}" or "rxn:{rxn_id}:{A}:{B}:{C}"
    parts = name.split(":")
    if len(parts) < 4:
        return None, None, None
    A = int(parts[2]); B = int(parts[3])
    C = int(parts[4]) if len(parts) > 4 else None
    return A, B, C

def _ids_from_pool(pool):
    return [x[0] for x in pool]

def generate_offspring_from_elites(rxn_id: int, n: int, elite_names: list[str],
                                   molecules_A, molecules_B, molecules_C, is_three_component: bool,
                                   mutation_prob: float = 0.1, seed: int | None = None,
                                   avoid_names: set[str] = None,
                                   avoid_inchikeys: set[str] = None,
                                   max_tries: int = 10) -> list[str]:
    rng = random.Random(seed) if seed is not None else random
    elite_As, elite_Bs, elite_Cs = set(), set(), set()
    for name in elite_names:
        A, B, C = _parse_components(name)
        if A is not None: elite_As.add(A)
        if B is not None: elite_Bs.add(B)
        if C is not None and is_three_component: elite_Cs.add(C)

    pool_A_ids = _ids_from_pool(molecules_A)
    pool_B_ids = _ids_from_pool(molecules_B)
    pool_C_ids = _ids_from_pool(molecules_C) if is_three_component else []

    out = []
    local_names = set()
    for _ in range(n):
        cand = None
        for _try in range(max_tries):
            use_mutA = (not elite_As) or (rng.random() < mutation_prob)
            use_mutB = (not elite_Bs) or (rng.random() < mutation_prob)
            use_mutC = (not elite_Cs) or (rng.random() < mutation_prob)

            A = rng.choice(pool_A_ids) if use_mutA else rng.choice(list(elite_As))
            B = rng.choice(pool_B_ids) if use_mutB else rng.choice(list(elite_Bs))
            if is_three_component:
                C = rng.choice(pool_C_ids) if use_mutC else rng.choice(list(elite_Cs))
                name = f"rxn:{rxn_id}:{A}:{B}:{C}"
            else:
                name = f"rxn:{rxn_id}:{A}:{B}"

            if avoid_names and name in avoid_names:
                continue
            if name in local_names:
                continue

            if avoid_inchikeys:
                try:
                    s = name_to_smiles_cached(name)
                    if s:
                        key = smiles_to_inchikey(s)
                        if key and key in avoid_inchikeys:
                            continue
                except Exception:
                    pass

            cand = name
            break

        if cand is None:
            cand = name
        out.append(cand)
        local_names.add(cand)
        if avoid_names is not None:
            avoid_names.add(cand)
    return out

@lru_cache(maxsize=200_000)
def smiles_to_inchikey(s: str) -> Optional[str]:
    try:
        mol = Chem.MolFromSmiles(s)
        if not mol:
            return None
        return Chem.MolToInchiKey(mol)
    except Exception:
        return None

@lru_cache(maxsize=200_000)
def name_to_smiles_cached(name: str) -> Optional[str]:
    try:
        return get_smiles_from_reaction(name)
    except Exception:
        return None

def run_sampler(n_samples: int = 1000, 
                seed: int = None, 
                subnet_config: dict = None, 
                output_path: str = None, 
                save_to_file: bool = False,
                db_path: str = None,
                elite_names: list[str] = None,
                elite_frac: float = 0.5,
                mutation_prob: float = 0.1,
                avoid_inchikeys: set[str] = None):
    reactions = get_available_reactions(db_path)
    if not reactions:
        bt.logging.error("No reactions found in the database, check db path and integrity.")
        return

    rxn_id = int(subnet_config["allowed_reaction"].split(":")[-1])
    bt.logging.info(f"Generating {n_samples} random molecules for reaction {rxn_id}")

    # Generate molecules with validation in batches for efficiency
    sampler_data = generate_valid_random_molecules_batch(
        rxn_id, n_samples, db_path, subnet_config, batch_size=200, seed=seed,
        elite_names=elite_names, elite_frac=elite_frac, mutation_prob=mutation_prob,
        avoid_inchikeys=avoid_inchikeys
        )

    if save_to_file:
        with open(output_path, "w") as f:
            json.dump(sampler_data, f, ensure_ascii=False, indent=2)

    return sampler_data


if __name__ == "__main__":
    run_sampler()
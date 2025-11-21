import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import sys
import json
import time
from collections import defaultdict
from typing import Dict, Set

import bittensor as bt
import pandas as pd
from rdkit import Chem
from pathlib import Path
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import nova_ph2

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PARENT_DIR)

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/output")

from nova_ph2.PSICHIC.wrapper import PsichicWrapper
from random_sampler import run_sampler

DB_PATH = str(Path(nova_ph2.__file__).resolve().parent / "combinatorial_db" / "molecules.sqlite")

WRITE_TIME_THRESHOLD_1 = 20 * 60
WRITE_TIME_THRESHOLD_2 = 25 * 60
WRITE_EARLY_INTERVAL = 2

def get_config(input_file: os.path = os.path.join(BASE_DIR, "input.json")):
    """
    Get config from input file
    """
    with open(input_file, "r") as f:
        d = json.load(f)
    config = {**d.get("config", {}), **d.get("challenge", {})}
    return config

def build_component_weights(top_pool: pd.DataFrame, rxn_id: int) -> Dict[str, Dict[int, float]]:
    """
    Build component weights based on scores of molecules containing them.
    Returns dict with 'A', 'B', 'C' keys mapping to {component_id: weight}
    """
    weights = {'A': defaultdict(float), 'B': defaultdict(float), 'C': defaultdict(float)}
    counts = {'A': defaultdict(int), 'B': defaultdict(int), 'C': defaultdict(int)}
    
    if top_pool.empty:
        return weights
    
    # Extract component IDs and scores
    for _, row in top_pool.iterrows():
        name = row['name']
        score = row['score']
        parts = name.split(":")
        if len(parts) >= 4:
            try:
                A_id = int(parts[2])
                B_id = int(parts[3])
                weights['A'][A_id] += max(0, score)  # Only positive contributions
                weights['B'][B_id] += max(0, score)
                counts['A'][A_id] += 1
                counts['B'][B_id] += 1
                
                if len(parts) > 4:
                    C_id = int(parts[4])
                    weights['C'][C_id] += max(0, score)
                    counts['C'][C_id] += 1
            except (ValueError, IndexError):
                continue
    
    # Normalize by count and add smoothing
    for role in ['A', 'B', 'C']:
        for comp_id in weights[role]:
            if counts[role][comp_id] > 0:
                weights[role][comp_id] = weights[role][comp_id] / counts[role][comp_id] + 0.1  # Smoothing
    
    return weights

def select_diverse_elites(top_pool: pd.DataFrame, n_elites: int, min_score_ratio: float = 0.7) -> pd.DataFrame:
    """
    Select diverse elite molecules: top by score, but ensure diversity in component space.
    """
    if top_pool.empty or n_elites <= 0:
        return pd.DataFrame()
    
    # Take top candidates (more than needed for diversity filtering)
    top_candidates = top_pool.head(min(len(top_pool), n_elites * 3))
    if len(top_candidates) <= n_elites:
        return top_candidates
    
    # Score threshold: at least min_score_ratio of max score
    max_score = top_candidates['score'].max()
    threshold = max_score * min_score_ratio
    candidates = top_candidates[top_candidates['score'] >= threshold]
    
    # Select diverse set: prefer molecules with different components
    selected = []
    used_components = {'A': set(), 'B': set(), 'C': set()}
    
    # First, add top scorer
    if not candidates.empty:
        top_idx = candidates.index[0]
        top_row = candidates.iloc[0]
        selected.append(top_idx)
        parts = top_row['name'].split(":")
        if len(parts) >= 4:
            try:
                used_components['A'].add(int(parts[2]))
                used_components['B'].add(int(parts[3]))
                if len(parts) > 4:
                    used_components['C'].add(int(parts[4]))
            except (ValueError, IndexError):
                pass
    
    # Then add diverse molecules
    for idx, row in candidates.iterrows():
        if len(selected) >= n_elites:
            break
        if idx in selected:
            continue
        
        parts = row['name'].split(":")
        if len(parts) >= 4:
            try:
                A_id = int(parts[2])
                B_id = int(parts[3])
                C_id = int(parts[4]) if len(parts) > 4 else None
                
                # Prefer molecules with new components
                is_diverse = (A_id not in used_components['A'] or 
                             B_id not in used_components['B'] or
                             (C_id is not None and C_id not in used_components['C']))
                
                if is_diverse or len(selected) < n_elites * 0.5:  # Always take some top ones
                    selected.append(idx)
                    used_components['A'].add(A_id)
                    used_components['B'].add(B_id)
                    if C_id is not None:
                        used_components['C'].add(C_id)
            except (ValueError, IndexError):
                # If parsing fails, just add it
                if len(selected) < n_elites:
                    selected.append(idx)
    
    # Fill remaining slots with top scorers
    for idx, row in candidates.iterrows():
        if len(selected) >= n_elites:
            break
        if idx not in selected:
            selected.append(idx)
    
    return candidates.loc[selected[:n_elites]] if selected else candidates.head(n_elites)

def iterative_sampling_loop(
    db_path: str,
    output_path: str,
    config: dict,
    save_all_scores: bool = False
) -> None:
    target_models = []
    antitarget_models = []

    for seq in config["target_sequences"]:
        wrapper = PsichicWrapper()
        wrapper.initialize_model(seq)
        target_models.append(wrapper)
    for seq in config["antitarget_sequences"]:
        wrapper = PsichicWrapper()
        wrapper.initialize_model(seq)
        antitarget_models.append(wrapper)

    n_samples = config["num_molecules"] * 5
    n_samples_first_iteration = n_samples if config["allowed_reaction"] == "rxn:5" else n_samples * 4

    top_pool = pd.DataFrame(columns=["name", "smiles", "InChIKey", "score"])
    seen_inchikeys = set()

    iteration = 0
    mutation_prob = 0.1
    elite_frac = 0.25
    prev_avg_score = None
    score_improvement_rate = 0.0
    rxn_id = int(config["allowed_reaction"].split(":")[-1])

    start_time = time.time()
    iter_start_time = time.time()
    
    while True:
        iteration += 1
        iter_start_time = time.time()
        
        # Build component weights from top pool for score-guided sampling
        component_weights = build_component_weights(top_pool, rxn_id) if not top_pool.empty else None
        
        # Select diverse elites (not just top by score)
        elite_df = select_diverse_elites(top_pool, min(100, len(top_pool))) if not top_pool.empty else pd.DataFrame()
        elite_names = elite_df["name"].tolist() if not elite_df.empty else None
        
        # Adaptive sampling: adjust based on score improvement
        if prev_avg_score is not None and not top_pool.empty:
            current_avg = top_pool['score'].mean()
            score_improvement_rate = (current_avg - prev_avg_score) / max(abs(prev_avg_score), 1e-6)
            
            # If improving well, increase exploitation; if stagnating, increase exploration
            if score_improvement_rate > 0.01:  # Good improvement
                elite_frac = min(0.7, elite_frac * 1.1)
                mutation_prob = max(0.05, mutation_prob * 0.95)
            elif score_improvement_rate < -0.01:  # Declining
                elite_frac = max(0.2, elite_frac * 0.9)
                mutation_prob = min(0.4, mutation_prob * 1.1)
        
        sampler_data = run_sampler(
            n_samples=n_samples_first_iteration if iteration == 1 else n_samples, 
            subnet_config=config, 
            db_path=db_path,
            elite_names=elite_names,
            elite_frac=elite_frac,
            mutation_prob=mutation_prob,
            avoid_inchikeys=seen_inchikeys,
            component_weights=component_weights,
        )
        
        if not sampler_data:
            bt.logging.warning("[Miner] No valid molecules produced; continuing")
            continue

        names = sampler_data["molecules"]
        smiles = sampler_data["smiles"]
        filtered_names = []
        filtered_smiles = []
        filtered_inchikeys = []
        
        # Pre-compute InChIKeys in batch for faster filtering
        inchikey_cache = {}
        for name, smile in zip(names, smiles):
            if not smile:
                continue
            try:
                # Check cache first
                if smile in inchikey_cache:
                    key = inchikey_cache[smile]
                else:
                    mol = Chem.MolFromSmiles(smile)
                    if not mol:
                        continue
                    key = Chem.MolToInchiKey(mol)
                    inchikey_cache[smile] = key
                
                if key in seen_inchikeys:
                    continue
                filtered_names.append(name)
                filtered_smiles.append(smile)
                filtered_inchikeys.append(key)
            except Exception:
                continue

        if not filtered_names:
            continue

        dup_ratio = (len(names) - len(filtered_names)) / max(1, len(names))
        if dup_ratio > 0.6:
            mutation_prob = min(0.5, mutation_prob * 1.5)
            elite_frac = max(0.2, elite_frac * 0.8)
        elif dup_ratio < 0.2 and not top_pool.empty:
            mutation_prob = max(0.05, mutation_prob * 0.9)
            elite_frac = min(0.8, elite_frac * 1.1)

        sampler_data = {"molecules": filtered_names, "smiles": filtered_smiles, "inchikeys": filtered_inchikeys}

        # Parallelize scoring across models (targets and antitargets)
        # Use larger batches for GPU efficiency (RTX 4090 can handle large batches)
        SCORE_BATCH_SIZE = 512  # Optimized for RTX 4090
        
        def score_with_model(model, batch_smiles):
            try:
                # Score in batches for GPU efficiency
                all_scores = []
                for i in range(0, len(batch_smiles), SCORE_BATCH_SIZE):
                    batch = batch_smiles[i:i + SCORE_BATCH_SIZE]
                    res = model.score_molecules(batch)
                    all_scores.extend(res['predicted_binding_affinity'].tolist())
                return all_scores
            except Exception as e:
                bt.logging.error(f"[Miner] Model scoring failed: {e}")
                # Return zeros to avoid breaking the iteration
                return [0.0] * len(batch_smiles)

        all_target_results = []
        all_antitarget_results = []

        # Score targets in parallel
        with ThreadPoolExecutor(max_workers=max(1, len(target_models))) as executor:
            futures = {executor.submit(score_with_model, m, filtered_smiles): idx for idx, m in enumerate(target_models)}
            for fut in as_completed(futures):
                all_target_results.append(fut.result())

        # Score antitargets in parallel
        with ThreadPoolExecutor(max_workers=max(1, len(antitarget_models))) as executor:
            futures = {executor.submit(score_with_model, m, filtered_smiles): idx for idx, m in enumerate(antitarget_models)}
            for fut in as_completed(futures):
                all_antitarget_results.append(fut.result())

        score_dict = {
            'ps_target_scores': all_target_results,
            'ps_antitarget_scores': all_antitarget_results,
        }

        # Calculate final scores per molecule
        batch_scores = calculate_final_scores(score_dict, sampler_data, config, save_all_scores)

        try:
            seen_inchikeys.update([k for k in batch_scores["InChIKey"].tolist() if k])
        except Exception:
            pass

        # Merge, deduplicate, sort and take top x
        top_pool = pd.concat([top_pool, batch_scores])
        top_pool = top_pool.drop_duplicates(subset=["InChIKey"], keep="first")
        top_pool = top_pool.sort_values(by="score", ascending=False)
        top_pool = top_pool.head(config["num_molecules"])
        
        # Track score improvement
        current_avg_score = top_pool['score'].mean() if not top_pool.empty else None
        if current_avg_score is not None:
            if prev_avg_score is not None:
                score_improvement_rate = (current_avg_score - prev_avg_score) / max(abs(prev_avg_score), 1e-6)
            prev_avg_score = current_avg_score

        # Conditional write based on elapsed time thresholds (atomic replace via temp file)
        elapsed = time.time() - start_time
        iter_time = time.time() - iter_start_time
        should_write = False
        if elapsed >= WRITE_TIME_THRESHOLD_2:
            should_write = True
        elif elapsed >= WRITE_TIME_THRESHOLD_1 and iteration % WRITE_EARLY_INTERVAL == 0:
            should_write = True

        if should_write and not top_pool.empty:
            top_entries = {"molecules": top_pool["name"].tolist()}
            tmp = output_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(top_entries, f, ensure_ascii=False, indent=2)
            os.replace(tmp, output_path)
            bt.logging.info(f"[Miner] Iteration {iteration} | Wrote {len(top_pool)} molecules to {output_path} (elapsed={elapsed:.1f}s, iter={iter_time:.1f}s)")
            bt.logging.info(f"[Miner] Average score: {top_pool['score'].mean():.4f} | Max: {top_pool['score'].max():.4f} | Improvement rate: {score_improvement_rate*100:.2f}%")
            bt.logging.info(f"[Miner] Elite frac: {elite_frac:.3f} | Mutation prob: {mutation_prob:.3f}")

def calculate_final_scores(score_dict: dict, 
        sampler_data: dict, 
        config: dict, 
        save_all_scores: bool = False,
        current_epoch: int = 0) -> pd.DataFrame:
    """
    Calculate final scores per molecule
    """

    names = sampler_data["molecules"]
    smiles = sampler_data["smiles"]
    inchikey_list = sampler_data.get("inchikeys")

    # Calculate InChIKey for each molecule only if not provided by caller
    if inchikey_list is None:
        inchikey_list = []
        for s in smiles:
            try:
                inchikey_list.append(Chem.MolToInchiKey(Chem.MolFromSmiles(s)))
            except Exception as e:
                bt.logging.error(f"Error calculating InChIKey for {s}: {e}")
                inchikey_list.append(None)

    # Calculate final scores for each molecule
    targets = score_dict['ps_target_scores']
    antitargets = score_dict['ps_antitarget_scores']

    # Vectorize score aggregation with NumPy for speed
    try:
        target_array = np.asarray(targets, dtype=np.float32)  # shape: (n_target_models, n_mols)
        antitarget_array = np.asarray(antitargets, dtype=np.float32)  # shape: (n_antitarget_models, n_mols)
        avg_target = target_array.mean(axis=0) if target_array.size else np.zeros(len(names), dtype=np.float32)
        avg_antitarget = antitarget_array.mean(axis=0) if antitarget_array.size else np.zeros(len(names), dtype=np.float32)
        final_scores = (avg_target - (config["antitarget_weight"] * avg_antitarget)).tolist()
    except Exception as e:
        bt.logging.error(f"[Miner] Vectorized score computation failed, falling back to Python loop: {e}")
        final_scores = []
        for mol_idx in range(len(names)):
            target_scores_for_mol = [target_list[mol_idx] for target_list in targets] if targets else [0.0]
            avg_t = sum(target_scores_for_mol) / len(target_scores_for_mol)
            antitarget_scores_for_mol = [antitarget_list[mol_idx] for antitarget_list in antitargets] if antitargets else [0.0]
            avg_at = sum(antitarget_scores_for_mol) / len(antitarget_scores_for_mol)
            final_scores.append(avg_t - (config["antitarget_weight"] * avg_at))

    # Store final scores in dataframe
    batch_scores = pd.DataFrame({
        "name": names,
        "smiles": smiles,
        "InChIKey": inchikey_list,
        "score": final_scores
    })

    # if save_all_scores:
    #     all_scores = {"scored_molecules": [(mol["name"], mol["score"]) for mol in batch_scores.to_dict(orient="records")]}
    #     all_scores_path = os.path.join(OUTPUT_DIR, f"all_scores_{current_epoch}.json")
    #     if os.path.exists(all_scores_path):
    #         with open(all_scores_path, "r") as f:
    #             all_previous_scores = json.load(f)
    #         all_scores["scored_molecules"] = all_previous_scores["scored_molecules"] + all_scores["scored_molecules"]
    #     with open(all_scores_path, "w") as f:
    #         json.dump(all_scores, f, ensure_ascii=False, indent=2)

    return batch_scores

def main(config: dict):
    iterative_sampling_loop(
        db_path=DB_PATH,
        output_path=os.path.join(OUTPUT_DIR, "result.json"),
        config=config,
        save_all_scores=True,
    )
 

if __name__ == "__main__":
    config = get_config()
    main(config)

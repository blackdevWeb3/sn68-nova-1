import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import sys
import json
import time

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
    with open(input_file, "r") as f:
        d = json.load(f)
    config = {**d.get("config", {}), **d.get("challenge", {})}
    return config

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

    start_time = time.time()
    while True:
        iteration += 1
        sampler_data = run_sampler(n_samples=n_samples_first_iteration if iteration == 1 else n_samples, 
                        subnet_config=config, 
                        db_path=db_path,
                        elite_names=top_pool["name"].tolist() if not top_pool.empty else None,
                        elite_frac=elite_frac,
                        mutation_prob=mutation_prob,
                        avoid_inchikeys=seen_inchikeys,
                        )
        
        if not sampler_data:
            bt.logging.warning("[Miner] No valid molecules produced; continuing")
            continue

        names = sampler_data["molecules"]
        smiles = sampler_data["smiles"]
        filtered_names = []
        filtered_smiles = []
        filtered_inchikeys = []
        for name, smile in zip(names, smiles):
            try:
                mol = Chem.MolFromSmiles(smile)
                if not mol:
                    continue
                key = Chem.MolToInchiKey(mol)
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
        def score_with_model(model, batch_smiles):
            try:
                res = model.score_molecules(batch_smiles)
                return res['predicted_binding_affinity'].tolist()
            except Exception as e:
                bt.logging.error(f"[Miner] Model scoring failed: {e}")
                # Return zeros to avoid breaking the iteration
                return [0.0] * len(batch_smiles)

        all_target_results = []
        all_antitarget_results = []

        with ThreadPoolExecutor(max_workers=max(1, len(target_models))) as executor:
            futures = {executor.submit(score_with_model, m, filtered_smiles): idx for idx, m in enumerate(target_models)}
            for fut in as_completed(futures):
                all_target_results.append(fut.result())

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

        # Conditional write based on elapsed time thresholds (atomic replace via temp file)
        elapsed = time.time() - start_time
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
            bt.logging.info(f"[Miner] Wrote {len(top_pool)} molecules to {output_path} (elapsed={elapsed:.1f}s)")
            bt.logging.info(f"[Miner] Average score: {top_pool['score'].mean()}")

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

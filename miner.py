import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import sys
import json

import bittensor as bt
import pandas as pd
from rdkit import Chem
from pathlib import Path
import nova_ph2

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PARENT_DIR)

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/output")

from nova_ph2.PSICHIC.wrapper import PsichicWrapper
from random_sampler import run_sampler

DB_PATH = str(Path(nova_ph2.__file__).resolve().parent / "combinatorial_db" / "molecules.sqlite")

def get_config(input_file: os.path = os.path.join(BASE_DIR, "input.json")):
    """
    Get config from input file
    """
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
    """
    Infinite loop, runs until orchestrator kills it:
      1) Sample n molecules
      2) Score them
      3) Merge with previous top x, deduplicate, sort, select top x
      4) Write top x to file (overwrite) each iteration
    """
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

        sampler_data = {"molecules": filtered_names, "smiles": filtered_smiles}

        all_target_results = []
        for target_model in target_models:
            target_results = target_model.score_molecules(filtered_smiles)
            all_target_results.append(target_results['predicted_binding_affinity'].tolist())
        all_antitarget_results = []
        for antitarget_model in antitarget_models:
            antitarget_results = antitarget_model.score_molecules(filtered_smiles)
            all_antitarget_results.append(antitarget_results['predicted_binding_affinity'].tolist())

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

        # format to accepted format
        top_entries = {"molecules": top_pool["name"].tolist()}

        # write to file
        with open(output_path, "w") as f:
            json.dump(top_entries, f, ensure_ascii=False, indent=2)

        bt.logging.info(f"[Miner] Wrote {config['num_molecules']} top molecules to {output_path}")
        bt.logging.info(f"[Miner] Average score: {top_pool['score'].mean()}")

def calculate_final_scores(score_dict: dict, 
        sampler_data: dict, 
        config: dict, 
        save_all_scores: bool = False,
        current_epoch: int = 0) -> pd.DataFrame:

    names = sampler_data["molecules"]
    smiles = sampler_data["smiles"]

    # Calculate InChIKey for each molecule to deduplicate molecules after merging
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
    final_scores = []
    for mol_idx in range(len(names)):
        # target average
        target_scores_for_mol = [target_list[mol_idx] for target_list in targets]
        avg_target = sum(target_scores_for_mol) / len(target_scores_for_mol)

        # antitarget average
        antitarget_scores_for_mol = [antitarget_list[mol_idx] for antitarget_list in antitargets]
        avg_antitarget = sum(antitarget_scores_for_mol) / len(antitarget_scores_for_mol)

        # final score
        score = avg_target - (config["antitarget_weight"] * avg_antitarget)
        final_scores.append(score)

    # Store final scores in dataframe
    batch_scores = pd.DataFrame({
        "name": names,
        "smiles": smiles,
        "InChIKey": inchikey_list,
        "score": final_scores
    })

    if save_all_scores:
        all_scores = {"scored_molecules": [(mol["name"], mol["score"]) for mol in batch_scores.to_dict(orient="records")]}
        all_scores_path = os.path.join(OUTPUT_DIR, f"all_scores_{current_epoch}.json")
        if os.path.exists(all_scores_path):
            with open(all_scores_path, "r") as f:
                all_previous_scores = json.load(f)
            all_scores["scored_molecules"] = all_previous_scores["scored_molecules"] + all_scores["scored_molecules"]
        with open(all_scores_path, "w") as f:
            json.dump(all_scores, f, ensure_ascii=False, indent=2)

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

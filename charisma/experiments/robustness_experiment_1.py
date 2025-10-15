#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import os
import random
import time
import json
from typing import Any, Dict, List, Tuple

import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from charisma.config import config
from charisma.interaction_generation.generation import run_interaction_pipeline
from charisma.evaluation.generation import evaluate_conversation_app

import logging
import datetime
import pytz
# Current working directory
current_dir = os.getcwd()
# Get absolute path
LOG_PATH = os.path.join(current_dir, "logs")
TIMESTAMP = str(datetime.datetime.now(pytz.timezone("Europe/Berlin"))).replace(" ", "_").replace(".", ":")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(f"{LOG_PATH}/{TIMESTAMP}.log"),
        logging.StreamHandler()
    ],
    force=True,
)

logging.info(f"Logging to {LOG_PATH}/{TIMESTAMP}.log")

load_dotenv()
OPEN_ROUTER_API_KEY = os.getenv("OPEN_ROUTER_API_KEY")
client = OpenAI(
    base_url="https://openrouter.ai/api/v1", api_key=OPEN_ROUTER_API_KEY
)

# ---------------- CSV helpers ----------------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def append_row_csv(path: str, row: Dict[str, Any], header: List[str]):
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in header})

def read_done_keys(results_csv: str) -> set:
    if not os.path.exists(results_csv):
        return set()
    df = pd.read_csv(results_csv, dtype={"condition_idx": int, "rep": int})
    return set((int(r.condition_idx), int(r.rep)) for _, r in df.iterrows())


# ---------------- Load inputs ----------------

def load_characters(characters_csv: str) -> pd.DataFrame:
    df = pd.read_csv(characters_csv)
    return df[["id", "mbti_profile"]].copy()

def load_scenarios(scen_csv: str, difficulty: str) -> pd.DataFrame:
    df = pd.read_csv(scen_csv)
    if "scenario_id" not in df.columns:
        df["scenario_id"] = [f"{difficulty}_idx_{i:04d}" for i in range(len(df))]
    df["difficulty"] = difficulty
    return df.copy()


# ---------------- Pairing ----------------

def round_robin_rounds(n: int) -> List[List[Tuple[int, int]]]:
    if n % 2 != 0:
        raise ValueError("Round-robin requires an even number of characters.")
    players = list(range(n))
    fixed = players[-1]
    rotating = players[:-1]
    rounds = []
    for _ in range(n - 1):
        left = [fixed] + rotating[: (n // 2 - 1)]
        right = list(reversed(rotating[(n // 2 - 1):]))
        pairs = [(left[i], right[i]) for i in range(n // 2)]
        rounds.append(pairs)
        rotating = [rotating[-1]] + rotating[:-1]
    return rounds

def build_pairs_round_robin(char_ids: List[str], k_rounds: int, rng: random.Random) -> List[Tuple[str, str]]:
    n = len(char_ids)
    rr = round_robin_rounds(n)
    idxs = list(range(len(rr)))
    rng.shuffle(idxs)
    chosen = idxs[:k_rounds]
    pairs_idx = [p for ridx in chosen for p in rr[ridx]]
    seen = set()
    pairs: List[Tuple[str, str]] = []
    for i, j in pairs_idx:
        a, b = sorted([char_ids[i], char_ids[j]])
        if (a, b) in seen:
            raise RuntimeError("Duplicate pair encountered.")
        seen.add((a, b))
        pairs.append((a, b))
    return pairs


def build_pairs_balanced_with_cap(
    char_ids: List[str],
    num_agent_pairs: int,
    cap: int,
    rng: random.Random,
    randomize_rounds: bool = True,
    randomize_feasible_p: bool = False,
) -> List[Tuple[str, str]]:
    """
    Build exactly `num_agent_pairs` pairs using a subset of `char_ids`,
    balancing appearances (each used agent appears R or R-1 times, <= cap),
    avoiding duplicate pairs, and making the selection random.
    """
    if cap < 1:
        raise ValueError("convos_per_agent (cap) must be >= 1")

    n = len(char_ids)
    P = num_agent_pairs
    total_slots = 2 * P
    capacity = n * cap
    if total_slots > capacity:
        raise ValueError(
            f"Infeasible: need {total_slots} agent-slots but capacity is {capacity} "
            f"(n={n}, cap={cap})."
        )

    import math

    # Find all feasible even p such that:
    #   R = ceil(P / (p/2)) <= cap  AND  R <= p-1  (can't exceed available RR rounds)
    feasible_ps = []
    min_p = max(2, math.ceil(2 * P / cap))
    if min_p % 2 == 1:
        min_p += 1
    for p in range(min_p, n + 1):
        if p % 2 == 1:
            continue
        pairs_per_round = p // 2
        R = math.ceil(P / pairs_per_round)
        if R <= cap and R <= (p - 1):
            feasible_ps.append((p, R, pairs_per_round))

    if not feasible_ps:
        raise ValueError(
            f"Infeasible: cannot schedule {P} pairs with cap={cap} using up to {n} agents."
        )

    # Choose p
    if randomize_feasible_p and len(feasible_ps) > 1:
        p, R, pairs_per_round = rng.choice(feasible_ps)
    else:
        p, R, pairs_per_round = feasible_ps[0]  # smallest p (most efficient)

    # Random subset of participants
    participants = rng.sample(char_ids, p)

    # Round-robin rounds over p participants
    rr_idx = round_robin_rounds(p)  # returns list of rounds of (i,j) index pairs
    if R > len(rr_idx):
        raise RuntimeError("Unexpected: requested more rounds than available.")

    idx2id = {i: a for i, a in enumerate(participants)}

    # Choose R rounds — randomly if requested
    round_ids = list(range(len(rr_idx)))
    if randomize_rounds:
        rng.shuffle(round_ids)
    chosen_round_ids = round_ids[:R]

    pairs: List[Tuple[str, str]] = []

    # Take R-1 full rounds
    for ridx in chosen_round_ids[:-1]:
        for i, j in rr_idx[ridx]:
            pairs.append(tuple(sorted((idx2id[i], idx2id[j]))))

    # Partial last round to hit exactly P (random subset of that round's disjoint pairs)
    need = P - len(pairs)
    last_round_pairs = rr_idx[chosen_round_ids[-1]][:]  # copy
    if randomize_rounds:
        rng.shuffle(last_round_pairs)
    for i, j in last_round_pairs[:need]:
        pairs.append(tuple(sorted((idx2id[i], idx2id[j]))))

    assert len(pairs) == P
    # No duplicates by construction (RR rounds are edge-disjoint across rounds)
    return pairs


# ---------------- Manifest ----------------

def make_assignment_header(num_repeats: int) -> List[str]:
    base = [
        "condition_idx", "pair_id", "agent_a_id", "agent_b_id", "agent_a_name", "agent_b_name",
        "scenario_id", "scenario", "difficulty",
    ]
    for r in range(num_repeats):
        base.append(f"rep{r}_seed")
    return base

RESULTS_COLS = [
    "condition_idx", "pair_id", "agent_a_id", "agent_b_id",
    "scenario_id",  "rep", "seed", "base_shared_goal", "social_goal_category", "explanation", 
    "shared_goal", "first_agent_goal", "second_agent_goal", "agent1_role", "agent2_role", "scenario",
    "difficulty", "agents", "interaction_history", 
    "evaluation_json", "status", "created_at",
]

def create_manifest(
    characters_csv: str,
    easy_csv: str,
    hard_csv: str,
    out_dir: str,
    global_seed: int,
    num_agent_pairs: int,
    convos_per_agent: int,
    num_repeats: int,
) -> pd.DataFrame:
    ensure_dir(out_dir)
    manifest_path = os.path.join(out_dir, "assignment_manifest.csv")
    if os.path.exists(manifest_path):
        return pd.read_csv(manifest_path)

    rng = random.Random(global_seed)

    # Characters
    chars_df = load_characters(characters_csv)
    char_ids = list(chars_df["id"].astype(str))
    n = len(char_ids)
    if n % 2 != 0:
        raise ValueError("Round-robin requires an even number of characters.")
    if 2 * num_agent_pairs != n * convos_per_agent:
        raise ValueError(f"Infeasible config: 2*num_agent_pairs={2*num_agent_pairs} "
                         f"must equal n*convos_per_agent={n*convos_per_agent}.")
    # Build pairs
    pairs = build_pairs_round_robin(char_ids, convos_per_agent, rng)
    if len(pairs) != num_agent_pairs:
        raise RuntimeError(f"Round-robin did not produce expected number of pairs {len(pairs)}.")

    # if n < 2:
    #     raise ValueError("Need at least 2 characters to form pairs.")

    # # --- NEW: balanced, cap-based pairing (no strict equality, no even-n requirement) ---
    # pairs = build_pairs_balanced_with_cap(
    #     char_ids=char_ids,
    #     num_agent_pairs=num_agent_pairs,
    #     cap=convos_per_agent,
    #     rng=rng,
    # )
    # # logging.info("hello", len(pairs), len(pairs_r))
    # logging.info(f"pairs {len(pairs)}")
    logging.info(pairs)
    # logging.info(pairs_r)

    # Scenarios
    easy_df = load_scenarios(easy_csv, "easy")
    hard_df = load_scenarios(hard_csv, "hard")
    if len(easy_df) + len(hard_df) < num_agent_pairs:
        raise ValueError("Not enough scenarios to assign.")

    want_easy = min(len(easy_df), num_agent_pairs // 2)
    want_hard = num_agent_pairs - want_easy
    easy_ids = list(zip(easy_df["scenario_id"], easy_df["scenario"]))
    hard_ids = list(zip(hard_df["scenario_id"], hard_df["scenario"]))
    rng.shuffle(easy_ids)
    rng.shuffle(hard_ids)
    chosen = [(sid, scen, "easy") for sid, scen in easy_ids[:want_easy]] + \
             [(sid, scen, "hard") for sid, scen in hard_ids[:want_hard]]
    rng.shuffle(chosen)

    rows = []
    for condition_idx, ((a, b), (sid, scen, diff)) in enumerate(zip(pairs, chosen)):
        pair_id = f"{a}__{b}"
        rep_seeds = [rng.randrange(1, 2**31 - 1) for _ in range(num_repeats)]
        char_a_name = chars_df[chars_df['id'] == int(a)].iloc[0]['mbti_profile']
        char_b_name = chars_df[chars_df['id'] == int(b)].iloc[0]['mbti_profile']
        row = {
            "condition_idx": condition_idx,
            "pair_id": pair_id,
            "agent_a_id": a,
            "agent_b_id": b,
            "agent_a_name": char_a_name,
            "agent_b_name": char_b_name,
            "scenario_id": sid,
            "scenario": scen,
            "difficulty": diff,
        }
        for r in range(num_repeats):
            row[f"rep{r}_seed"] = rep_seeds[r]
        rows.append(row)

    manifest = pd.DataFrame(rows, columns=make_assignment_header(num_repeats))
    manifest.to_csv(manifest_path, index=False)
    logging.info(f"[manifest] wrote {manifest_path}")
    return manifest


# ---------------- Runner ----------------

def run_experiment(
    characters_filepath: str,
    easy_scenario_csv: str,
    hard_scenario_csv: str,
    behavioral_coding_csv: str,
    out_dir: str,
    global_seed: int,
    num_agent_pairs: int,
    convos_per_agent: int,
    num_repeats: int,
    model: str,
    provider: str = None,
):
    ensure_dir(out_dir)
    results_csv = os.path.join(out_dir, "results.csv")


    easy_df = load_scenarios(easy_scenario_csv, "easy")
    hard_df = load_scenarios(hard_scenario_csv, "hard")

    manifest = create_manifest(
        characters_filepath, easy_scenario_csv, hard_scenario_csv, out_dir,
        global_seed, num_agent_pairs, convos_per_agent, num_repeats,
    )

    done = read_done_keys(results_csv)
    total = len(manifest) * num_repeats
    completed = 0

    for _, row in manifest.iterrows():
        cond_idx = int(row["condition_idx"])
        pair_id = str(row["pair_id"])
        a_id = str(row["agent_a_id"])
        b_id = str(row["agent_b_id"])
        sid = str(row["scenario_id"])
        scen_text = str(row["scenario"])
        diff = str(row["difficulty"])


        for rep in range(num_repeats):
            if (cond_idx, rep) in done:
                completed += 1
                continue

            seed = int(row[f"rep{rep}_seed"])

            # load the scenario row
            if diff == "easy":
                scen_row = easy_df.loc[easy_df["scenario_id"] == sid].iloc[0]
            else:
                scen_row = hard_df.loc[hard_df["scenario_id"] == sid].iloc[0]
            
            # Convert to dict if you want
            scenario_data = scen_row.to_dict()


            # load agents
            chars_df = load_characters(characters_filepath)

            agent_1 = chars_df.loc[chars_df["id"] == int(row["agent_a_id"])].iloc[0].to_dict()
            agent_2 = chars_df.loc[chars_df["id"] == int(row["agent_b_id"])].iloc[0].to_dict()
            agent1_name = agent_1['mbti_profile']
            agent2_name = agent_2['mbti_profile']
            agents_list = [agent1_name, agent2_name]
            logging.info(f"characters {agents_list}")

            status = "ok"
            try:
                interaction_output = run_interaction_pipeline(
                    behavioral_coding_filename=behavioral_coding_csv,
                    agent1_name=agent1_name,
                    agent2_name=agent2_name,
                    scenario_data=scenario_data,
                    model=model,
                    provider=provider,
                    max_turns=20,
                    num_samples=None
                )
            except Exception as e:
                interaction_output = []
                status = "interaction_failed"
                raise e
            
            logging.info(f"\ninteraction_output: {interaction_output}")

            evaluation_output = {}
            if status == "ok":
                try:
                    scenario_setting = {
                        "shared_goal": scenario_data["shared_goal"],
                        "chosen_goal_category": scenario_data["social_goal_category"],
                        "first_agent_goal": scenario_data["first_agent_goal"],
                        "second_agent_goal": scenario_data["second_agent_goal"],
                        "first_agent_role": scenario_data["agent1_role"],
                        "second_agent_role": scenario_data["agent2_role"],
                    }
                    interaction = [{"agent": d["agent"], "response": d["response"]} for d in interaction_output]
                    evaluation_output = evaluate_conversation_app(scenario_setting, interaction, client, model, provider)
                except Exception as e:
                    evaluation_output = {"error": str(e)}
                    status = "evaluation_failed"
                    raise e
            logging.info(f"evaluation_output: {evaluation_output}")

            # Write result row
            row_out = {
                "condition_idx": cond_idx,
                "pair_id": pair_id,
                "agent_a_id": a_id,
                "agent_b_id": b_id,
                "scenario_id": sid,
                "agents": f"{agents_list}",
                "base_shared_goal": scenario_data["base_shared_goal"],
                "social_goal_category": scenario_data["social_goal_category"],
                "explanation": scenario_data["explanation"],
                "shared_goal": scenario_data["shared_goal"],
                "first_agent_goal": scenario_data["first_agent_goal"],
                "second_agent_goal": scenario_data["second_agent_goal"],
                "agent1_role": scenario_data["agent1_role"],
                "agent2_role": scenario_data["agent2_role"],
                "scenario": scenario_data["scenario"],
                "interaction_history":  json.dumps(interaction_output),
                "difficulty": diff,
                "rep": rep,
                "seed": seed,
                "evaluation_json": json.dumps(evaluation_output, ensure_ascii=False, separators=(",", ":")),
                "status": status,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            append_row_csv(results_csv, row_out, RESULTS_COLS)
            completed += 1
            pct = 100.0 * completed / total
            logging.info(f"[{completed}/{total}] cond={cond_idx} rep={rep} -> {status} ({pct:.1f}%)")

    logging.info(f"Done. Results: {results_csv}")
    logging.info(f"Manifest: {os.path.join(out_dir, 'assignment_manifest.csv')}")


# ---------------- Entrypoint ----------------

if __name__ == "__main__":
    logging.info("config", config)
    cfg = config.pipeline.experiments.robustness_1
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

    run_experiment(
        characters_filepath=os.path.join(root, cfg.characters_filepath),
        easy_scenario_csv=os.path.join(root, cfg.easy_scenario_csv),
        hard_scenario_csv=os.path.join(root, cfg.hard_scenario_csv),
        behavioral_coding_csv=os.path.join(root,  config.pipeline.behavioral_coding_csv),
        out_dir=os.path.join(root, "outputs", "robustness_1"),
        global_seed=42,
        num_agent_pairs=cfg.num_agent_pairs,
        convos_per_agent=cfg.agent_per_pair,
        num_repeats=cfg.num_repeats,
        model=config.pipeline.model,
        provider=config.pipeline.provider if config.pipeline.provider else None,
    )

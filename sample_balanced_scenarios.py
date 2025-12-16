import pandas as pd
from collections import Counter


def balanced_sample(df, category_col: str, target_total: int, random_state: int = 42):
    """
    Sample 'target_total' rows from df, trying to keep a balanced distribution
    over 'category_col'. Works in two steps:
      1. Give each category up to 2 rows (baseline), if available.
      2. Distribute remaining rows to categories that still have capacity
         (more rows available), prioritizing those with the most capacity.
    This is done *within* a single dataframe.
    """
    counts = df[category_col].value_counts()

    # Step 1: Baseline of up to 2 per category
    baseline = {cat: min(2, counts[cat]) for cat in counts.index}
    baseline_sum = sum(baseline.values())

    if baseline_sum > target_total:
        # Fallback: if 2-per-category is already too many, just scale down
        # by sampling uniformly later.
        baseline = {cat: 0 for cat in counts.index}
        baseline_sum = 0

    leftover = max(0, target_total - baseline_sum)

    # Step 2: Distribute leftover
    extra = {cat: 0 for cat in counts.index}
    # Categories ordered by remaining capacity
    order = sorted(counts.index, key=lambda c: counts[c] - baseline[c], reverse=True)

    i = 0
    while leftover > 0 and any(counts[c] - baseline[c] - extra[c] > 0 for c in counts.index):
        cat = order[i % len(order)]
        if counts[cat] - baseline[cat] - extra[cat] > 0:
            extra[cat] += 1
            leftover -= 1
        i += 1

    target_sizes = {cat: baseline[cat] + extra[cat] for cat in counts.index}

    parts = []
    for cat, n in target_sizes.items():
        if n <= 0:
            continue
        sub = df[df[category_col] == cat]
        n_eff = min(n, len(sub))
        if n_eff > 0:
            parts.append(sub.sample(n=n_eff, random_state=random_state))

    sampled = pd.concat(parts, ignore_index=True)

    # Adjust size if we overshoot or undershoot
    if len(sampled) > target_total:
        sampled = sampled.sample(n=target_total, random_state=random_state)
    elif len(sampled) < target_total:
        remaining = df.loc[~df.index.isin(sampled.index)]
        need = target_total - len(sampled)
        extra_rows = remaining.sample(n=need, random_state=random_state)
        sampled = pd.concat([sampled, extra_rows], ignore_index=True)

    return sampled


def sample_easy_and_hard_unique(
    easy_csv: str,
    hard_csv: str,
    output_csv: str = "deepseek__scenarios_EasyHard_balanced_45.csv",
    easy_total: int = 20,
    hard_total: int = 25,
    category_col: str = "social_goal_category",
    shared_goal_col: str = "shared_goal",
    random_state: int = 42,
):
    # Load data
    df_easy = pd.read_csv(easy_csv)
    df_hard = pd.read_csv(hard_csv)

    # Basic sanity checks
    for col in [category_col, shared_goal_col]:
        if col not in df_easy.columns:
            raise ValueError(f"Column '{col}' not found in Easy CSV.")
        if col not in df_hard.columns:
            raise ValueError(f"Column '{col}' not found in Hard CSV.")

    if easy_total > len(df_easy):
        raise ValueError(f"Requested {easy_total} Easy rows but file has only {len(df_easy)}.")
    if hard_total > len(df_hard):
        raise ValueError(f"Requested {hard_total} Hard rows but file has only {len(df_hard)}.")

    # ---- 1) Sample Easy, balanced within Easy ----
    df_easy_sample = balanced_sample(df_easy, category_col, easy_total, random_state=random_state)

    # ---- 2) Filter Hard so that shared_goal does not overlap with Easy ----
    used_goals = set(df_easy_sample[shared_goal_col].unique())
    df_hard_filtered = df_hard[~df_hard[shared_goal_col].isin(used_goals)].copy()

    if len(df_hard_filtered) < hard_total:
        raise RuntimeError(
            f"After excluding Easy shared goals, only {len(df_hard_filtered)} Hard rows remain, "
            f"but you requested {hard_total}. You may need to lower hard_total."
        )

    # ---- 3) Sample Hard from the filtered set, balanced within Hard ----
    df_hard_sample = balanced_sample(df_hard_filtered, category_col, hard_total, random_state=random_state)

    # ---- 4) Combine & annotate ----
    df_easy_sample["difficulty"] = "Easy"
    df_hard_sample["difficulty"] = "Hard"

    df_combined = pd.concat([df_easy_sample, df_hard_sample], ignore_index=True)

    # Final sanity checks
    if len(df_combined) != easy_total + hard_total:
        raise RuntimeError(
            f"Combined samples should have {easy_total + hard_total} rows "
            f"but got {len(df_combined)}."
        )

    # Ensure uniqueness of shared_goal across Easy & Hard
    easy_goals = set(df_easy_sample[shared_goal_col])
    hard_goals = set(df_hard_sample[shared_goal_col])
    overlap = easy_goals & hard_goals
    if overlap:
        raise RuntimeError(
            f"Found overlap in shared_goal between Easy and Hard: {len(overlap)} overlaps."
        )

    # Save result
    df_combined.to_csv(output_csv, index=False)

    # Print some info
    print(f"Saved combined {len(df_combined)} rows to: {output_csv}\n")

    print("Overall social_goal_category distribution:")
    print(df_combined[category_col].value_counts(), "\n")

    print("Easy vs Hard by social_goal_category:")
    print(pd.crosstab(df_combined[category_col], df_combined["difficulty"]))

    return df_combined

import pandas as pd
from collections import Counter


def balanced_sample(df, category_col: str, target_total: int, random_state: int = 42):
    """
    Sample 'target_total' rows from df, trying to keep a balanced distribution
    over 'category_col'. Works in two steps:
      1. Give each category up to 2 rows (baseline), if available.
      2. Distribute remaining rows to categories that still have capacity
         (more rows available), prioritizing those with the most capacity.
    This is done *within* a single dataframe.
    """
    counts = df[category_col].value_counts()

    # Step 1: baseline of up to 2 per category
    baseline = {cat: min(2, counts[cat]) for cat in counts.index}
    baseline_sum = sum(baseline.values())

    if baseline_sum > target_total:
        # If 2-per-category is already too many, drop baseline and let
        # the later step and final fix handle balancing.
        baseline = {cat: 0 for cat in counts.index}
        baseline_sum = 0

    leftover = max(0, target_total - baseline_sum)

    # Step 2: distribute leftover to categories with remaining capacity
    extra = {cat: 0 for cat in counts.index}
    # Order categories by remaining capacity
    order = sorted(counts.index, key=lambda c: counts[c] - baseline[c], reverse=True)

    i = 0
    while leftover > 0 and any(counts[c] - baseline[c] - extra[c] > 0 for c in counts.index):
        cat = order[i % len(order)]
        if counts[cat] - baseline[cat] - extra[cat] > 0:
            extra[cat] += 1
            leftover -= 1
        i += 1

    target_sizes = {cat: baseline[cat] + extra[cat] for cat in counts.index}

    # Actually sample per category
    parts = []
    for cat, n in target_sizes.items():
        if n <= 0:
            continue
        sub = df[df[category_col] == cat]
        n_eff = min(n, len(sub))
        if n_eff > 0:
            parts.append(sub.sample(n=n_eff, random_state=random_state))

    sampled = pd.concat(parts, ignore_index=True)

    # Fix size if off (slight rounding / capacity issues)
    if len(sampled) > target_total:
        sampled = sampled.sample(n=target_total, random_state=random_state)
    elif len(sampled) < target_total:
        remaining = df.loc[~df.index.isin(sampled.index)]
        need = target_total - len(sampled)
        extra_rows = remaining.sample(n=need, random_state=random_state)
        sampled = pd.concat([sampled, extra_rows], ignore_index=True)

    return sampled


def sample_hard_then_easy_unique(
    easy_csv: str,
    hard_csv: str,
    output_csv: str = "deepseek__scenarios_EasyHard_balanced_45.csv",
    hard_total: int = 25,
    easy_total: int = 20,
    category_col: str = "social_goal_category",
    shared_goal_col: str = "shared_goal",
    random_state: int = 42,
):
    # Load data
    df_easy = pd.read_csv(easy_csv)
    df_hard = pd.read_csv(hard_csv)

    # Sanity checks
    for col in [category_col, shared_goal_col]:
        if col not in df_easy.columns:
            raise ValueError(f"Column '{col}' not found in Easy CSV.")
        if col not in df_hard.columns:
            raise ValueError(f"Column '{col}' not found in Hard CSV.")

    if hard_total > len(df_hard):
        raise ValueError(f"Requested {hard_total} Hard rows but file has only {len(df_hard)}.")
    if easy_total > len(df_easy):
        raise ValueError(f"Requested {easy_total} Easy rows but file has only {len(df_easy)}.")

    # ---- 1) First: sample Hard, balanced within Hard ----
    df_hard_sample = balanced_sample(
        df_hard,
        category_col=category_col,
        target_total=hard_total,
        random_state=random_state,
    )

    # ---- 2) Remove used shared goals from Easy ----
    used_goals = set(df_hard_sample[shared_goal_col].unique())
    df_easy_filtered = df_easy[~df_easy[shared_goal_col].isin(used_goals)].copy()

    if len(df_easy_filtered) < easy_total:
        raise RuntimeError(
            f"After excluding Hard shared goals, only {len(df_easy_filtered)} Easy rows remain, "
            f"but you requested {easy_total}. You may need to lower easy_total."
        )

    # ---- 3) Sample Easy from remaining rows, balanced within Easy ----
    df_easy_sample = balanced_sample(
        df_easy_filtered,
        category_col=category_col,
        target_total=easy_total,
        random_state=random_state,
    )

    # ---- 4) Combine & check ----
    df_hard_sample["difficulty"] = "Hard"
    df_easy_sample["difficulty"] = "Easy"

    df_combined = pd.concat([df_hard_sample, df_easy_sample], ignore_index=True)

    # Size check
    expected_total = hard_total + easy_total
    if len(df_combined) != expected_total:
        raise RuntimeError(
            f"Combined samples should have {expected_total} rows but got {len(df_combined)}."
        )

    # Uniqueness check
    hard_goals = set(df_hard_sample[shared_goal_col])
    easy_goals = set(df_easy_sample[shared_goal_col])
    overlap = hard_goals & easy_goals
    if overlap:
        raise RuntimeError(
            f"Found overlap in shared_goal between Hard and Easy: {len(overlap)} overlaps."
        )

    # Save
    df_combined.to_csv(output_csv, index=False)

    # Optional: print distributions
    print(f"Saved combined {len(df_combined)} rows to: {output_csv}\n")

    print("Overall social_goal_category distribution:")
    print(df_combined[category_col].value_counts(), "\n")

    print("Easy vs Hard by social_goal_category:")
    print(pd.crosstab(df_combined[category_col], df_combined["difficulty"]))

    return df_combined


def shuffle_csv(df: pd.DataFrame, random_state: int = 42, save_path: str = None):
    """
    Shuffle a DataFrame in a reproducible way.
    If save_path is provided, saves the shuffled DataFrame to CSV.
    """
    shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    if save_path:
        shuffled.to_csv(save_path, index=False)

    return shuffled

if __name__ == "__main__":
    # sample_hard_then_easy_unique(
    #     easy_csv="outputs/deepseek__scenarios_Easy.csv",
    #     hard_csv="outputs/deepseek__scenarios_Hard.csv",
    #     output_csv="outputs/deepseek__scenarios_balanced_45_hard_then_easy.csv",
    #     hard_total=25,
    #     easy_total=20,
    #     category_col="social_goal_category",
    #     shared_goal_col="shared_goal",
    #     random_state=42,
    # )
    shuffle_csv(
        pd.read_csv("outputs/deepseek__scenarios_balanced_45.csv"),
        random_state=42,
        save_path="outputs/deepseek__scenarios_balanced_45_shuffled.csv",
    )

# if __name__ == "__main__":
#     sample_easy_and_hard_unique(
#         easy_csv="outputs/deepseek__scenarios_Easy.csv",
#         hard_csv="outputs/deepseek__scenarios_Hard.csv",
#         output_csv="outputs/deepseek__scenarios_balanced_45.csv",
#         easy_total=20,
#         hard_total=25,
#         category_col="social_goal_category",
#         shared_goal_col="shared_goal",
#         random_state=42,
#     )

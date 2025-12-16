import os
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols


# ---------- helpers ----------

def extract_scenario_type(folder_name: str) -> str:
    if folder_name.endswith("DEEPSEEK_SCENARIOS"):
        return "DEEPSEEK"
    if folder_name.endswith("GPT_SCENARIOS"):
        return "GPT"
    return "OTHER"


def extract_eval_model(folder_name: str) -> str:
    name = folder_name
    if name.endswith("_SCENARIOS"):
        name = name[:-len("_SCENARIOS")]

    if "_evaluation_visualisations_" in name:
        tail = name.split("_evaluation_visualisations_", 1)[1]
        eval_model = tail.rsplit("-", 1)[0]
        return eval_model

    return "UNKNOWN"


def extract_interaction_model(folder_name: str) -> str:
    if "_evaluation" in folder_name:
        return folder_name.split("_evaluation", 1)[0]
    return folder_name

def spearman_for_scenario_given_interaction_and_eval(all_ranks,
                                                     interaction_model,
                                                     eval_model):
    """
    Computes Spearman correlation between ScenarioTypes (DEEPSEEK vs GPT),
    given a fixed InteractionModel and a fixed EvalModel.
    """

    print(f"\n=== Spearman between ScenarioTypes (Interaction={interaction_model}, Eval={eval_model}) ===\n")

    # Filter to specific interaction model + evaluator
    df = all_ranks[
        (all_ranks["InteractionModel"] == interaction_model) &
        (all_ranks["EvalModel"] == eval_model)
    ]

    # Pivot: rows = Agents, columns = ScenarioType, values = Rank
    pivot = df.pivot_table(
        index="Agent",
        columns="ScenarioType",
        values="Rank",
        aggfunc="mean"
    )

    # Need at least 2 scenario types (DEEPSEEK + GPT)
    if pivot.shape[1] < 2:
        print(f"Not enough scenario types for InteractionModel={interaction_model} and EvalModel={eval_model}.")
        return None

    spearman_corr = pivot.corr(method="spearman")

    print(spearman_corr.round(3))

    spearman_corr.to_csv(
        f"spearman_scenario/spearman_scenarios_given_{interaction_model}_{eval_model}.csv"
    )

    return spearman_corr

def spearman_for_interaction_given_eval(all_ranks, eval_model="deepseekv3.1", scenario="DEEPSEEK"):
    """
    Computes Spearman correlation between InteractionModels,
    keeping EvalModel fixed and ScenarioType fixed.
    """

    print(f"\n=== Spearman similarity between InteractionModels (EvalModel={eval_model}, ScenarioType={scenario}) ===\n")

    # filter by evaluation model + scenario type
    df = all_ranks[
        (all_ranks["EvalModel"] == eval_model) &
        (all_ranks["ScenarioType"] == scenario)
    ]

    # pivot: rows = Agent, columns = InteractionModels, values = Rank
    pivot = df.pivot_table(
        index="Agent",
        columns="InteractionModel",
        values="Rank",
        aggfunc="mean"
    )

    # compute Spearman correlation between interaction models
    corr = pivot.corr(method="spearman")

    print(corr.round(3))

    corr.to_csv(f"spearman_interactionmodels_given_{eval_model}.csv")

    return corr

def compute_spearman_per_interaction(all_ranks, chosen_scenario="DEEPSEEK"):
    """
    Computes Spearman rank similarity between evaluation models,
    separately for each InteractionModel, restricted to one ScenarioType.
    """

    # filter scenario type
    df = all_ranks[all_ranks["ScenarioType"] == chosen_scenario]

    interaction_models = df["InteractionModel"].unique()

    results = {}

    print(f"\n=== Spearman Similarity Between EvalModels per InteractionModel (ScenarioType={chosen_scenario}) ===\n")

    for im in interaction_models:
        sub = df[df["InteractionModel"] == im]

        # pivot: rows = agents, cols = eval models, values = ranks
        pivot = sub.pivot_table(
            index="Agent",
            columns="EvalModel",
            values="Rank",
            aggfunc="mean"
        )

        if pivot.shape[1] < 2:
            print(f"Skipping {im}: only one evaluation model present.")
            continue

        spearman_corr = pivot.corr(method="spearman")

        results[im] = spearman_corr

        print(f"\n--- InteractionModel: {im} ---")
        print(spearman_corr.round(3))

        # optionally save
        spearman_corr.to_csv(f"spearman_evalmodel_for_{im}_{chosen_scenario}.csv")

    return results
# ---------- main aggregation ----------

def main(root_dir: str = "."):
    rows = []

    for entry in os.scandir(root_dir):
        if not entry.is_dir():
            continue

        folder = entry.name
        csv_path = os.path.join(entry.path, "agent_performance_results.csv")
        if not os.path.isfile(csv_path):
            continue

        scenario_type = extract_scenario_type(folder)
        eval_model = extract_eval_model(folder)
        interaction_model = extract_interaction_model(folder)

        df = pd.read_csv(csv_path)

        if "Personal_Score" not in df.columns:
            raise ValueError(f"'Personal_Score' missing in {csv_path}")
        if "Agent" not in df.columns:
            raise ValueError(f"'Agent' missing in {csv_path}")

        ranks = df["Personal_Score"].rank(ascending=False, method="average")

        for agent, rank, score in zip(df["Agent"], ranks, df["Personal_Score"]):
            rows.append(
                {
                    "Agent": agent,
                    "Rank": rank,
                    "Personal_Score": score,
                    "Folder": folder,
                    "ScenarioType": scenario_type,
                    "EvalModel": eval_model,
                    "InteractionModel": interaction_model,
                }
            )

    all_ranks = pd.DataFrame(rows)
    # all_ranks.to_csv("all_agent_ranks_personal_scores.csv", index=False)
    deepseek_only = all_ranks[(all_ranks["EvalModel"] == "deepseekv3.1") & (all_ranks["ScenarioType"] == "DEEPSEEK")]

    # Compute average rank per interaction model under DeepSeek evaluation
    avg_rank_deepseek = (
        deepseek_only.groupby(["InteractionModel", "Agent"])
        .agg(AvgRank=("Rank", "mean"), AvgSharedScore=("Personal_Score", "mean"))
        .sort_values(["InteractionModel", "AvgRank"])
        .reset_index()
    )

    # avg_rank_deepseek.to_csv("avg_rank_for_deepseek_eval.csv", index=False)
    print("\n=== Average placement for each model (DeepSeek evaluation only) ===")
    print(avg_rank_deepseek)
    if all_ranks.empty:
        print("No data found.")
        return

    # ------------ 1. AGGREGATIONS ------------

    def aggregate_and_save(df, group_cols, filename):
        out = (
            df.groupby(group_cols)
            .agg(
                AvgRank=("Rank", "mean"),
                AvgPersonalScore=("Personal_Score", "mean")
            )
            .sort_values(group_cols)
            .reset_index()
        )
        out.to_csv(filename, index=False)
        return out

    # avg_all = aggregate_and_save(all_ranks, ["Agent"], "avg_rank_and_score_all_agents_shared.csv")
    # avg_scenario = aggregate_and_save(all_ranks, ["ScenarioType", "Agent"], "avg_rank_and_score_by_scenario_shared.csv")
    # avg_eval = aggregate_and_save(all_ranks, ["EvalModel", "Agent"], "avg_rank_and_score_by_eval_model_shared.csv")
    # avg_interaction = aggregate_and_save(all_ranks, ["InteractionModel", "Agent"], "avg_rank_and_score_by_interaction_model_shared.csv")

    # ------------ 2. VARIANCE DECOMPOSITION (ANOVA) ------------

    print("\n========== Running Variance Decomposition (ANOVA) ==========\n")

    model = ols(
        "Personal_Score ~ C(EvalModel) + C(InteractionModel) + C(ScenarioType)",
        data=all_ranks
    ).fit()

    anova_table = sm.stats.anova_lm(model, typ=2)

    # Add variance explained
    anova_table["VarianceExplained"] = anova_table["sum_sq"] / anova_table["sum_sq"].sum()

    print(anova_table)
    print("\n========== Variance Explained (%) ==========\n")
    print((anova_table["VarianceExplained"] * 100).round(2))

    # Save ANOVA results
    # anova_table.to_csv("variance_decomposition_anova.csv")

    #   compute_spearman_per_interaction(all_ranks, chosen_scenario="GPT")
    # spearman_for_interaction_given_eval(all_ranks, eval_model="deepseekv3.1", scenario="DEEPSEEK")
    interaction_models = all_ranks["InteractionModel"].unique()

    for im in interaction_models:
        # spearman_for_scenario_given_interaction_and_eval(
        #     all_ranks,
        #     interaction_model=im,
        #     eval_model="deepseekv3.1"
        # )

        spearman_for_scenario_given_interaction_and_eval(
            all_ranks,
            interaction_model=im,
            eval_model="gpt5-mini"
        )

if __name__ == "__main__":
    main(".")

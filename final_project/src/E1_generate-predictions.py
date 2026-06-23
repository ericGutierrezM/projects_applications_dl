#!/usr/bin/env python3
"""
Infrastructure Construction Trajectory Analysis Pipeline.

This script processes construction delta metrics for both a baseline method and
a pre-trained Vision Transformer (ChangeStar) model. It cleans data, computes
cumulative construction metrics, classifies completion trajectories, flags anomalies,
and evaluates reporting discrepancies against ground-truth government data.
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

def analyze_construction_trajectory(df_input, delta_cols, prefix=""):
    """
    Processes construction delta metrics, classifies trajectories, performs consistency 
    checks against project status, visualizes high-risk discrepancies, plots trend lines,
    and prints an analytical executive summary.
    """
    # Work on a copy to avoid SettingWithCopyWarning
    df = df_input.copy()
    
    # 1. Clamp negative model noise
    df[delta_cols] = df[delta_cols].clip(lower=0)

    # 2. Cumulative sums with custom prefix
    cum_suffixes = ["cum_25", "cum_50", "cum_75", "cum_100", "cum_post"]
    cum_cols = [f"{prefix}{suffix}" for suffix in cum_suffixes]
    
    cum = df[delta_cols].cumsum(axis=1)
    cum.columns = cum_cols
    df = pd.concat([df, cum], axis=1)

    # 3. Calculate S_t metrics
    total_change_col = f"{prefix}total_change"
    df[total_change_col] = df[f"{prefix}cum_post"]

    eps = 1e-8
    s_suffixes = ["S_25", "S_50", "S_75", "S_100"]
    
    for s_suff in s_suffixes:
        df[f"{prefix}{s_suff}"] = df[f"{prefix}{s_suff.lower().replace('s', 'cum')}"] / (df[total_change_col] + eps)
    df[f"{prefix}S_post"] = 1.0

    print(f"\n--- Dataframe after calculating S_t (Prefix: '{prefix}') ---")
    print(df.head(2))

    # 4. State Sequence generation
    def score_to_state(s):
        if s < 0.15: return "N"
        elif s < 0.70: return "C"
        return "F"

    trajectory_suffixes = ["S_25", "S_50", "S_75", "S_100", "S_post"]
    state_cols = []
    
    for suff in trajectory_suffixes:
        col_name = f"{prefix}{suff}"
        state_col = col_name.replace("S_", "state_")
        state_cols.append(state_col)
        df[state_col] = df[col_name].apply(score_to_state)

    df[f"{prefix}state_sequence"] = df[state_cols].agg("-".join, axis=1)
    
    print("\n--- Dataframe after State Sequence mapping ---")
    print(df[[f"{prefix}state_sequence"]].head(5))

    # 5. Trajectory Classification
    def classify_trajectory(row):
        s25 = row[f"{prefix}S_25"]
        s50 = row[f"{prefix}S_50"]
        s75 = row[f"{prefix}S_75"]
        s100 = row[f"{prefix}S_100"]
        trajectory = [s25, s50, s75, s100]

        if max(trajectory) < 0.15: return "Never Started"
        if s25 > 0.70: return "Fast Completion"
        if s50 > 0.30 and (s100 - s50) < 0.15: return "Stalled"
        if s75 < 0.30 and s100 > 0.80: return "Completed Weirdly"
        return "Normal Completion"

    traj_type_col = f"{prefix}trajectory_type"
    df[traj_type_col] = df.apply(classify_trajectory, axis=1)

    print("\n--- Trajectory Type Value Counts ---")
    print(df[traj_type_col].value_counts())
    
    if "status" in df.columns:
        print("\n--- Project Status Value Counts ---")
        print(df["status"].value_counts())

    # 6. Consistency Labeling
    def consistency_label(row):
        if "status" not in row:
            return "Unknown"
        status = str(row["status"]).lower()
        traj = row[traj_type_col]

        if "completed" in status:
            return "Consistent" if traj in ["Normal Completion", "Fast Completion"] else "Potential Discrepancy"
        if "ongoing" in status:
            if traj == "Stalled": return "Consistent"
            return "Possibly Ahead" if traj == "Normal Completion" else "Review"
        if "terminated" in status:
            return "Consistent" if traj in ["Stalled", "Never Started"] else "Potential Discrepancy"
        return "Unknown"

    consistency_col = f"{prefix}consistency"
    df[consistency_col] = df.apply(consistency_label, axis=1)

    print("\n--- Consistency Distribution ---")
    print(df[consistency_col].value_counts())

    if "status" in df.columns:
        print("\n--- Status vs Trajectory Cross-tabulation ---")
        print(pd.crosstab(df["status"], df[traj_type_col], margins=True))

    # 7. High Risk Subset Exploration
    high_risk = df[df[consistency_col] == "Potential Discrepancy"].copy()
    print(f'\nTotal High Risk Records Found: {len(high_risk)}')

    display_cols = ["contractId", "status", traj_type_col, f"{prefix}S_25", f"{prefix}S_50", f"{prefix}S_75", f"{prefix}S_100"]
    existing_display_cols = [c for c in display_cols if c in high_risk.columns]
    print(high_risk[existing_display_cols].head(20))

    # 8. Image Visualization Block
    def display_project(row):
        folder = row.get("image_folder_path", "")
        patterns = {
            "00": "*_00_pre_start.png", "25": "*_25_progress.png", "50": "*_50_progress.png",
            "75": "*_75_progress.png", "100": "*_100_completed.png", "post": "*_post_completion.png",
            "mask": "*_road_mask.png"
        }

        fig = plt.figure(figsize=(24, 8))
        gs = fig.add_gridspec(2, 7, height_ratios=[3, 1])

        delta_str = "\n    ".join([f"Δ{i+1} ({col}) = {row[col]:.4f}" for i, col in enumerate(delta_cols)])

        title = f"""
        Project: {row.get('contractId', row.get('contractID', 'N/A'))}
        Description: {row.get('description', 'N/A')}
        Status: {row.get('status', 'N/A')} | Taxonomy: {row[traj_type_col]} | Consistency: {row[consistency_col]}
        
        {delta_str}
        """
        fig.suptitle(title, fontsize=13)

        for idx, (label, pattern) in enumerate(patterns.items()):
            ax = fig.add_subplot(gs[0, idx])
            matches = glob.glob(os.path.join(folder, pattern)) if folder else []
            if len(matches) > 0:
                img = Image.open(matches[0])
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, "Missing", ha="center", va="center")
            ax.set_title(label)
            ax.axis("off")

        ax2 = fig.add_subplot(gs[1, :])
        delta_vals = [row[col] for col in delta_cols]
        labels = [f"P{i+1}" for i in range(len(delta_cols))] if len(delta_cols) != 5 else ["0→25", "25→50", "50→75", "75→100", "100→Post"]

        ax2.bar(labels, delta_vals)
        ax2.set_ylabel("Changed ROI Fraction")
        ax2.set_title("Observed Construction Activity")
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()

    if not high_risk.empty:
        for taxonomy in high_risk[traj_type_col].unique():
            print("\n" + "="*80)
            print(f"VISUALIZING SAMPLE FOR: {taxonomy}")
            print("="*80)

            subset = high_risk[high_risk[traj_type_col] == taxonomy]
            sampled_subset = subset.sample(min(5, len(subset)), random_state=42)

            for _, row in sampled_subset.iterrows():
                display_project(row)

    # 9. Cohort Trend Plots (Sample of 20 per Taxonomy)
    print("\n--- Generating Trajectory Performance Plots ---")
    taxonomy_examples = df.groupby(traj_type_col).head(20)

    for traj in taxonomy_examples[traj_type_col].unique():
        subset = taxonomy_examples[taxonomy_examples[traj_type_col] == traj]
        
        plt.figure(figsize=(8, 5))
        for _, row in subset.iterrows():
            plt.plot(
                [25, 50, 75, 100, 110],
                [
                    row[f"{prefix}S_25"],
                    row[f"{prefix}S_50"],
                    row[f"{prefix}S_75"],
                    row[f"{prefix}S_100"],
                    row[f"{prefix}S_post"]
                ],
                alpha=0.2
            )
        plt.title(f"Cohort Profile: {traj}")
        plt.ylabel("Observed Completion Score ($S_t$)")
        plt.xlabel("Project Timeline (%)")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.show()

    # 10. Final Analytical Summary Output
    n_projects = len(df)
    traj_counts = df[traj_type_col].value_counts(normalize=True).mul(100).round(1)
    consistency_counts = df[consistency_col].value_counts(normalize=True).mul(100).round(1)

    print("\n" + "=" * 60)
    print("INFRASTRUCTURE TRAJECTORY ANALYSIS EXECUTIVE REPORT")
    print("=" * 60)
    print(f"\nProjects analyzed: {n_projects:,}")
    
    print("\nTrajectory Distribution:")
    for k, v in traj_counts.items():
        print(f"  {k}: {v}%")

    print("\nConsistency Assessment:")
    for k, v in consistency_counts.items():
        print(f"  {k}: {v}%")
    print("=" * 60 + "\n")

    return df


def main():
    # Define file paths
    baseline_path = 'data/dataset_w_baseline.csv'
    changestar_path = 'data/dataset_w_changestar.csv'
    output_predictions_path = 'data/dataset_predictions.csv'

    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)

    print("Loading data variants...")
    baseline_raw_df = pd.read_csv(baseline_path)
    changestar_raw_df = pd.read_csv(changestar_path)

    # Base baseline configuration
    baseline_cols = ["delta_00_to_25", "delta_25_to_50", "delta_50_to_75", "delta_75_to_10", "delta_10_to_po"]
    
    # ChangeStar configuration
    changestar_cols = [
        'vit_delta_00_to_25_t0.5', 
        'vit_delta_25_to_50_t0.5', 
        'vit_delta_50_to_75_t0.5',
        'vit_delta_75_to_10_t0.5', 
        'vit_delta_10_to_po_t0.5'
    ]

    # --- Step 1 & 3: Process and Print Baseline ---
    print("\n" + "="*80)
    print("RUNNING ANALYSIS ON BASELINE MODEL EXPERIMENT")
    print("="*80)
    processed_base_df = analyze_construction_trajectory(
        df_input=baseline_raw_df,
        delta_cols=baseline_cols,
        prefix="base_"
    )

    # --- Step 2 & 3: Process and Print Vision Transformer (ChangeStar) ---
    print("\n" + "="*80)
    print("RUNNING ANALYSIS ON PRE-TRAINED CHANGESTAR (ViT) MODEL EXPERIMENT")
    print("="*80)
    processed_vit_df = analyze_construction_trajectory(
        df_input=changestar_raw_df,
        delta_cols=changestar_cols,
        prefix="vit_"
    )

    # --- Step 4: Merge DataFrames and Save Predictions ---
    print("\nMerging datasets and saving predictions...")
    
    # Isolate newly engineered vit_ metrics columns along with contractId
    vit_new_cols = [col for col in processed_vit_df.columns if col.startswith("vit_")]
    vit_subset = processed_vit_df[['contractId'] + vit_new_cols]

    # Combine metrics on master contract identifier
    merged_predictions_df = pd.merge(processed_base_df, vit_subset, on='contractId', how='left')
    merged_predictions_df.to_csv(output_predictions_path, index=False)
    print(f"Merged output successfully written to: {output_predictions_path}")

    # --- Step 5: Print Discrepancy ID Comparison Report ---
    # Query IDs marked as high-risk anomalies by checking respective prefix frameworks
    vit_discrepancies = set(merged_predictions_df.query('vit_consistency == "Potential Discrepancy"').contractId.unique())
    base_discrepancies = set(merged_predictions_df.query('base_consistency == "Potential Discrepancy"').contractId.unique())

    # Calculate set alignments
    common_ids = vit_discrepancies & base_discrepancies
    only_in_vit = vit_discrepancies - base_discrepancies
    only_in_base = base_discrepancies - vit_discrepancies
    all_unique_ids = vit_discrepancies ^ base_discrepancies

    # Format and present discrepancy analysis report
    print("\n" + "=" * 60)
    print("DISCREPANCY ID COMPARISON REPORT")
    print("=" * 60)
    print(f"Total ViT Discrepancies:  {len(vit_discrepancies)}")
    print(f"Total Base Discrepancies: {len(base_discrepancies)}")
    print("-" * 60)

    print(f"✅ Common IDs (found in BOTH frameworks): {len(common_ids)}")
    if common_ids:
        print(f"  Sample: {list(common_ids)[:5]}")

    print(f"\n🔍 Unique to ViT Framework Flags: {len(only_in_vit)}")
    if only_in_vit:
        print(f"  Sample: {list(only_in_vit)[:5]}")

    print(f"\n🔍 Unique to Base Framework Flags: {len(only_in_base)}")
    if only_in_base:
        print(f"  Sample: {list(only_in_base)[:5]}")

    print(f"\n💥 Total Distinct Symmetric Unique IDs (In either, but not both): {len(all_unique_ids)}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
import pandas as pd

# =========================
# LOAD CSVs
# =========================

ttest = pd.read_csv("t_test_results.csv")
ablation = pd.read_csv("ablation_t_test_results.csv")

# =========================
# T-TEST LATEX TABLE
# =========================

with open("table_ttest.tex", "w") as f:
    f.write("\\begin{table}[h!]\n\\centering\n")
    f.write("\\caption{Welch t-test results between Classical A* and Learned A*}\n")
    f.write("\\begin{tabular}{lcccc}\n")
    f.write("\\hline\n")
    f.write("Metric & Mean A & Mean B & p-value & Significant \\\\\n")
    f.write("\\hline\n")

    for _, row in ttest.iterrows():
        f.write(f"{row['metric']} & "
                f"{row['mean_A']:.2f} & "
                f"{row['mean_B']:.2f} & "
                f"{row['p_value']:.2e} & "
                f"{'Yes' if row['significant(p<0.05)'] else 'No'} \\\\\n")

    f.write("\\hline\n\\end{tabular}\n\\end{table}\n")

# =========================
# ABLATION LATEX TABLE
# =========================

with open("table_ablation.tex", "w") as f:
    f.write("\\begin{table}[h!]\n\\centering\n")
    f.write("\\caption{Ablation study with Welch t-test significance}\n")
    f.write("\\begin{tabular}{l l c c c}\n")
    f.write("\\hline\n")
    f.write("Comparison & Metric & Mean Full & Mean Ablation & p-value \\\\\n")
    f.write("\\hline\n")

    for _, row in ablation.iterrows():
        f.write(f"{row['comparison']} & "
                f"{row['metric']} & "
                f"{row['mean_full']:.2f} & "
                f"{row['mean_ablation']:.2f} & "
                f"{row['p_value']:.2e} \\\\\n")

    f.write("\\hline\n\\end{tabular}\n\\end{table}\n")

print("✅ LaTeX tables generated:")
print(" - table_ttest.tex")
print(" - table_ablation.tex")

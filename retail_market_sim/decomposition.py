"""Price-volume-mix decomposition and market-share calculations."""
import numpy as np
import pandas as pd
from .data import CELL_KEYS


def _pack_size_to_factor(pack_size: str) -> float:
    value = pd.to_numeric(pd.Series([pack_size]), errors="coerce").iloc[0]
    if pd.isna(value) or not np.isfinite(value) or value <= 0:
        raise ValueError(f"Pack_Size must be a finite positive litres-per-pack value; got {pack_size!r}.")
    return float(value)


def _ensure_equiv_volumes(scenario: pd.DataFrame) -> pd.DataFrame:
    if "Equiv_Volume_Baseline" not in scenario.columns:
        scenario = scenario.copy()
        scenario["Equiv_Volume_Baseline"] = scenario.apply(lambda row: _pack_size_to_factor(row["Pack_Size"]) * row["Baseline_Unit_Sales"], axis=1)
        scenario["Equiv_Volume_Simulated"] = scenario.apply(lambda row: _pack_size_to_factor(row["Pack_Size"]) * row["Simulated_Unit_Sales"], axis=1)
    return scenario


def three_factor_decomposition(scenario: pd.DataFrame, group_columns: list) -> pd.DataFrame:
    scenario = _ensure_equiv_volumes(scenario)
    if not group_columns:
        scenario = scenario.copy()
        scenario["_total_market"] = 1
        group_columns = ["_total_market"]
        drop_total = True
    else:
        drop_total = False
    g = scenario.groupby(group_columns, as_index=False, observed=True).agg(
        Q0=("Equiv_Volume_Baseline", "sum"), Q1=("Equiv_Volume_Simulated", "sum"), R0=("Baseline_Revenue", "sum"), R1=("Simulated_Revenue", "sum"))
    if (g["Q0"] <= 0).any() or (g["Q1"] <= 0).any():
        raise ValueError("Each decomposition group requires positive total equivalized volume in both baseline and scenario.")
    detail = scenario.copy()
    detail = detail.merge(g[group_columns + ["Q1"]], on=group_columns, how="left", validate="many_to_one")
    detail["s0"] = detail["Equiv_Volume_Baseline"] / detail.groupby(group_columns)["Equiv_Volume_Baseline"].transform("sum")
    detail["s1"] = detail["Equiv_Volume_Simulated"] / detail.groupby(group_columns)["Equiv_Volume_Simulated"].transform("sum")
    detail["P0_equiv"] = detail["Baseline_Revenue"] / detail["Equiv_Volume_Baseline"]
    detail["P1_equiv"] = detail["Simulated_Revenue"] / detail["Equiv_Volume_Simulated"]
    g["Pbar0"] = g["R0"] / g["Q0"]
    g["Volume_Effect"] = (g["Q1"] - g["Q0"]) * g["Pbar0"]
    mix = detail.groupby(group_columns, as_index=False, observed=True).apply(
        lambda df: pd.Series({"Mix_Effect": df["Q1"].iloc[0] * ((df["s1"] - df["s0"]) * df["P0_equiv"]).sum()}), include_groups=False).reset_index()
    g = g.merge(mix, on=group_columns, how="left")
    price = detail.groupby(group_columns, as_index=False, observed=True).apply(
        lambda df: pd.Series({"Price_Effect": (df["Equiv_Volume_Simulated"] * (df["P1_equiv"] - df["P0_equiv"])).sum()}), include_groups=False).reset_index()
    g = g.merge(price, on=group_columns, how="left")
    g["Decomposition_Sum"] = g["Volume_Effect"] + g["Mix_Effect"] + g["Price_Effect"]
    g["Revenue_Delta"] = g["R1"] - g["R0"]
    error = g["Decomposition_Sum"] - g["Revenue_Delta"]
    if not np.allclose(error, 0.0, rtol=1e-9, atol=1e-6):
        raise RuntimeError(f"Decomposition reconciliation failed: max error = {error.abs().max():.6f}")
    if drop_total:
        g = g.drop(columns=["_total_market"])
    return g


def value_shares(scenario: pd.DataFrame, group_columns: list, parent_columns: list | None = None) -> pd.DataFrame:
    grouped = scenario.groupby(group_columns, as_index=False, observed=True).agg(Baseline_Revenue=("Baseline_Revenue", "sum"), Simulated_Revenue=("Simulated_Revenue", "sum"))
    if parent_columns:
        baseline_denominator = grouped.groupby(parent_columns, observed=True)["Baseline_Revenue"].transform("sum")
        simulated_denominator = grouped.groupby(parent_columns, observed=True)["Simulated_Revenue"].transform("sum")
    else:
        baseline_denominator = grouped["Baseline_Revenue"].sum()
        simulated_denominator = grouped["Simulated_Revenue"].sum()
    grouped["Baseline_Share_Pct"] = grouped["Baseline_Revenue"] / baseline_denominator * 100
    grouped["Simulated_Share_Pct"] = grouped["Simulated_Revenue"] / simulated_denominator * 100
    grouped["Share_Change_pp"] = grouped["Simulated_Share_Pct"] - grouped["Baseline_Share_Pct"]
    return grouped.replace([np.inf, -np.inf], np.nan)

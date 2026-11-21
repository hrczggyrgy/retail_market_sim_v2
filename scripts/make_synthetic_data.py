#!/usr/bin/env python
"""
Generate synthetic beverage market data with known seasonal behavior and price responses.

Used for smoke tests and synthetic recovery checks.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def generate_synthetic_data(
    n_months: int = 24,
    n_categories: int = 3,
    n_brands_per_cat: int = 2,
    n_skus_per_brand: int = 2,
    n_retailers: int = 3,
    seed: int = 42,
    price_response: float = -1.5,
    seasonality_strength: float = 0.1,
    trend: float = 0.0,
    noise_scale: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate synthetic beverage market data.
    
    Parameters
    ----------
    n_months : Number of months of data
    n_categories : Number of categories
    n_brands_per_cat : Brands per category
    n_skus_per_brand : SKUs per brand (different pack sizes)
    n_retailers : Number of retailers
    seed : Random seed for reproducibility
    price_response : True price elasticity (negative)
    seasonality_strength : Amplitude of seasonal effect
    trend : Monthly trend in log units
    noise_scale : Observation noise scale
    
    Returns
    -------
    sales_df : Monthly sales data
    sku_master_df : SKU metadata
    """
    rng = np.random.default_rng(seed)
    
    # Build SKU hierarchy
    categories = [f"Category_{i+1}" for i in range(n_categories)]
    brands = {}
    skus = {}
    pack_sizes = [0.25, 0.33, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    
    for cat in categories:
        brands[cat] = [f"{cat}_Brand_{j+1}" for j in range(n_brands_per_cat)]
        for brand in brands[cat]:
            skus[brand] = []
            for k in range(n_skus_per_brand):
                sku_id = f"{brand}_SKU_{k+1}"
                skus[brand].append(sku_id)
    
    retailers = [f"Retailer_{r+1}" for r in range(n_retailers)]
    
    # True parameters
    # Category-level intercepts
    cat_intercepts = {cat: rng.normal(8.0, 0.5) for cat in categories}
    
    # Brand-level effects within category
    brand_effects = {}
    for cat in categories:
        for brand in brands[cat]:
            brand_effects[brand] = rng.normal(0.0, 0.3)
    
    # SKU-level effects within brand
    sku_effects = {}
    for brand in skus:
        for sku in skus[brand]:
            sku_effects[sku] = rng.normal(0.0, 0.2)
    
    # Retailer effects
    retailer_effects = {r: rng.normal(0.0, 0.15) for r in retailers}
    
    # Price elasticity (partially pooled)
    # True elasticity varies by category
    cat_elasticity = {cat: rng.normal(price_response, 0.2) for cat in categories}
    # Brand deviation within category
    brand_elast_dev = {}
    for cat in categories:
        for brand in brands[cat]:
            brand_elast_dev[brand] = rng.normal(0.0, 0.15)
    # SKU deviation within brand
    sku_elast_dev = {}
    for brand in skus:
        for sku in skus[brand]:
            sku_elast_dev[sku] = rng.normal(0.0, 0.1)
    
    # Seasonal effects (monthly)
    seasonal = {cat: {m: rng.normal(0.0, seasonality_strength) for m in range(1, 13)} for cat in categories}
    # Center seasonals to sum to zero per category
    for cat in categories:
        mean_seas = np.mean(list(seasonal[cat].values()))
        for m in seasonal[cat]:
            seasonal[cat][m] -= mean_seas
    
    # Base prices (log scale) per SKU–retailer
    base_log_price = {}
    for sku in sku_effects:
        for r in retailers:
            base_log_price[(sku, r)] = rng.normal(0.5, 0.2)
    
    # Price variation over time (with some retailer-SKU specific shocks)
    price_shocks = {}
    for t in range(1, n_months + 1):
        for sku in sku_effects:
            for r in retailers:
                price_shocks[(t, sku, r)] = rng.normal(0.0, 0.05)
    
    # Generate monthly data
    rows = []
    sku_master_rows = []
    
    for t in range(1, n_months + 1):
        month = pd.Timestamp(f"2024-01-01") + pd.DateOffset(months=t-1)
        month_str = month.strftime("%Y-%m-%d")
        season = (t - 1) % 12 + 1
        
        for cat in categories:
            cat_int = cat_intercepts[cat]
            cat_elast = cat_elasticity[cat]
            
            for brand in brands[cat]:
                b_eff = brand_effects[brand]
                b_elast = brand_elast_dev[brand]
                
                for sku in skus[brand]:
                    s_eff = sku_effects[sku]
                    s_elast = sku_elast_dev[sku]
                    
                    # True elasticity for this SKU
                    true_elast = cat_elast + b_elast + s_elast
                    
                    for r in retailers:
                        r_eff = retailer_effects[r]
                        b_log_p = base_log_price[(sku, r)]
                        p_shock = price_shocks[(t, sku, r)]
                        log_p = b_log_p + p_shock
                        price = np.exp(log_p)
                        
                        # Category total demand
                        log_mu = (
                            cat_int + b_eff + s_eff + r_eff 
                            + true_elast * log_p 
                            + seasonal[cat][season]
                            + trend * t / 12.0
                        )
                        
                        # Generate quantity with noise
                        mu = np.exp(log_mu)
                        quantity = rng.poisson(max(1, mu))
                        
                        # Sales = quantity * price
                        sales = quantity * price
                        
                        # Pack size
                        pack_liters = rng.choice(pack_sizes)
                        
                        rows.append({
                            "Month": month_str,
                            "Category": cat,
                            "Retailer": r,
                            "Brand": brand,
                            "SKU": sku,
                            "Pack_Size": pack_liters,
                            "Unit_Sales": quantity,
                            "Sales_Revenue": round(sales, 2),
                            "Price": round(price, 4),
                        })
                        
                        # Track for SKU master
                        sku_master_rows.append({
                            "sku_id": sku,
                            "category_id": cat,
                            "brand_id": brand,
                            "pack_liters": pack_liters,
                            "sku_name": f"{brand} {sku}",
                        })
    
    # Deduplicate SKU master
    sku_master_df = pd.DataFrame(sku_master_rows).drop_duplicates(subset=["sku_id"]).reset_index(drop=True)
    sales_df = pd.DataFrame(rows)
    
    # Sort
    sales_df = sales_df.sort_values(["Month", "Category", "Retailer", "Brand", "SKU"]).reset_index(drop=True)
    
    return sales_df, sku_master_df


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic beverage market data")
    parser.add_argument("--months", type=int, default=24, help="Number of months")
    parser.add_argument("--categories", type=int, default=3, help="Number of categories")
    parser.add_argument("--brands-per-cat", type=int, default=2, help="Brands per category")
    parser.add_argument("--skus-per-brand", type=int, default=2, help="SKUs per brand")
    parser.add_argument("--retailers", type=int, default=3, help="Number of retailers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--price-response", type=float, default=-1.5, help="True price elasticity")
    parser.add_argument("--seasonality", type=float, default=0.1, help="Seasonality strength")
    parser.add_argument("--trend", type=float, default=0.0, help="Monthly trend in log units")
    parser.add_argument("--noise", type=float, default=0.1, help="Observation noise")
    parser.add_argument("--output-dir", type=str, default="data/examples", help="Output directory")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating synthetic data: {args.months} months, {args.categories} categories, "
          f"{args.brands_per_cat} brands/cat, {args.skus_per_brand} SKUs/brand, "
          f"{args.retailers} retailers, seed={args.seed}")
    
    sales_df, sku_master_df = generate_synthetic_data(
        n_months=args.months,
        n_categories=args.categories,
        n_brands_per_cat=args.brands_per_cat,
        n_skus_per_brand=args.skus_per_brand,
        n_retailers=args.retailers,
        seed=args.seed,
        price_response=args.price_response,
        seasonality_strength=args.seasonality,
        trend=args.trend,
        noise_scale=args.noise,
    )
    
    sales_path = output_dir / "synthetic_sales_monthly.csv"
    sku_path = output_dir / "synthetic_sku_master.csv"
    
    sales_df.to_csv(sales_path, index=False)
    sku_master_df.to_csv(sku_path, index=False)
    
    print(f"Generated {len(sales_df)} sales rows, {len(sku_master_df)} unique SKUs")
    print(f"Sales saved to: {sales_path}")
    print(f"SKU master saved to: {sku_path}")
    
    # Print summary
    print("\nCategories:", sales_df["Category"].unique().tolist())
    print("Retailers:", sales_df["Retailer"].unique().tolist())
    print("Months:", sorted(sales_df["Month"].unique()))
    print(f"Total units: {sales_df['Unit_Sales'].sum():,.0f}")
    print(f"Total revenue: {sales_df['Sales_Revenue'].sum():,.2f}")


if __name__ == "__main__":
    main()
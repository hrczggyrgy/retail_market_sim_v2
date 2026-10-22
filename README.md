# Retail Market Simulation v2

A POS-only framework for retail market modeling, elasticity estimation, and price-volume decomposition.

## Overview

This project provides a comprehensive mathematical framework for analyzing retail market data using only point-of-sale (POS) data. It covers:

- **Hierarchical aggregation** (SKU → Brand → Retailer → Category → Total Market)
- **Market share calculations** (value and volume shares)
- **Growth dynamics** (MoM, YoY, log growth)
- **Price elasticity modeling** (own-price, cross-price, arc elasticity)
- **Cannibalization analysis**
- **Price-volume-mix decomposition**
- **Seasonality adjustment**
- **Effect identification map** for POS-only data

## Key Files

| File | Description |
|------|-------------|
| `retail_market_model.md` | Complete mathematical specification with LaTeX equations |
| `sample_data.csv` | Example POS data structure |

## Mathematical Framework

The model defines notation and identities for:

1. **Fundamental Identities** — Revenue, price, equivalized volume
2. **Hierarchical Aggregation** — Revenue-weighted prices at every level
3. **Market Shares** — Value and volume shares by brand/retailer
4. **Growth Dynamics** — MoM, YoY, and log growth rates
5. **Share-Growth Decomposition** — Brand growth = Category growth + Share shift
6. **Elasticity Models** — Log-log regression specifications for own-price and cross-price elasticity
7. **Price-Volume Decomposition** — Standard two-factor (price/volume) plus mix effect
8. **Seasonality** — Multiplicative seasonal indices
9. **Effect Map** — What's identifiable from POS data alone

## Usage

The `retail_market_model.md` serves as a reference specification for implementing retail analytics pipelines. Equations are formatted in LaTeX for direct use in documentation, notebooks, or academic papers.

## Data Requirements

Minimum POS data fields needed:
- `period` (month)
- `category`, `retailer`, `brand`, `sku`
- `units` (quantity sold)
- `revenue` (value sales)
- `pack_size` (for equivalization)

## Citation

If you use this framework, please cite appropriately. The model follows standard retail analytics methodology (IRI/Nielsen conventions).

## License

MIT License
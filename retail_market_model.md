# Retail Market Model — POS-Only Framework

## 1. Notation and Data Structure

Let:

- $t = 1,\dots,24$ : calendar month
- $c$ : category
- $r$ : retailer
- $b$ : brand
- $s$ : SKU
- $Q_{t,c,r,b,s}$ : unit sales
- $R_{t,c,r,b,s}$ : revenue (value sales)
- $P_{t,c,r,b,s}$ : price per unit
- $w_s$ : pack-size conversion factor to a standardized volume unit
- $V_{t,c,r,b,s}$ : equivalized volume
- $\mathcal{S}_{c,r,b}$ : set of SKUs for brand $b$ in retailer $r$ and category $c$
- $\mathcal{B}_{c,r}$ : set of brands in retailer $r$ and category $c$
- $\mathcal{R}_c$ : set of retailers in category $c$
- $\mathcal{C}$ : set of categories in the total market

The hierarchy is:

$$
\text{Total Market} \;\to\; c \;\to\; r \;\to\; b \;\to\; s
$$

---

## 2. Fundamental Identities

### Revenue and Price

At SKU level:

$$
R_{t,c,r,b,s} = P_{t,c,r,b,s} \times Q_{t,c,r,b,s}
$$

Price per unit is the implied average selling price:

$$
P_{t,c,r,b,s} = \frac{R_{t,c,r,b,s}}{Q_{t,c,r,b,s}}
$$

### Equivalized Volume

Equivalized volume converts heterogeneous pack sizes into a common physical measure (kilograms, litres, or pieces):

$$
V_{t,c,r,b,s} = w_s \times Q_{t,c,r,b,s}
$$

**Examples:**

| SKU Description | Standard Unit | $w_s$ |
|-----------------|---------------|-------|
| 500 g SKU       | kilograms     | 0.5   |
| 1.5 L SKU       | litres        | 1.5   |
| 3-piece SKU     | pieces        | 3     |

Price per standardized volume unit:

$$
P^{V}_{t,c,r,b,s}
= \frac{R_{t,c,r,b,s}}{V_{t,c,r,b,s}}
= \frac{P_{t,c,r,b,s}}{w_s}
$$

This metric is essential when comparing SKUs with different pack sizes.

---

## 3. Hierarchical Aggregation

### SKU to Brand

$$
\begin{aligned}
Q_{t,c,r,b} &= \sum_{s \in \mathcal{S}_{c,r,b}} Q_{t,c,r,b,s} \\
R_{t,c,r,b} &= \sum_{s \in \mathcal{S}_{c,r,b}} R_{t,c,r,b,s} \\
V_{t,c,r,b} &= \sum_{s \in \mathcal{S}_{c,r,b}} V_{t,c,r,b,s}
\end{aligned}
$$

### Brand to Retailer

$$
\begin{aligned}
Q_{t,c,r} &= \sum_{b \in \mathcal{B}_{c,r}} Q_{t,c,r,b} \\
R_{t,c,r} &= \sum_{b \in \mathcal{B}_{c,r}} R_{t,c,r,b} \\
V_{t,c,r} &= \sum_{b \in \mathcal{B}_{c,r}} V_{t,c,r,b}
\end{aligned}
$$

### Retailer to Category

$$
\begin{aligned}
Q_{t,c} &= \sum_{r \in \mathcal{R}_c} Q_{t,c,r} \\
R_{t,c} &= \sum_{r \in \mathcal{R}_c} R_{t,c,r} \\
V_{t,c} &= \sum_{r \in \mathcal{R}_c} V_{t,c,r}
\end{aligned}
$$

### Category to Total Market

$$
\begin{aligned}
Q_t &= \sum_{c \in \mathcal{C}} Q_{t,c} \\
R_t &= \sum_{c \in \mathcal{C}} R_{t,c} \\
V_t &= \sum_{c \in \mathcal{C}} V_{t,c}
\end{aligned}
$$

### Revenue-Weighted Aggregate Price

Aggregate price must always be calculated as revenue divided by units, **not** as an unweighted average of SKU prices:

$$
\begin{aligned}
P_{t,c,r,b} &= \frac{R_{t,c,r,b}}{Q_{t,c,r,b}} \\
P_{t,c,r}   &= \frac{R_{t,c,r}}{Q_{t,c,r}} \\
P_{t,c}     &= \frac{R_{t,c}}{Q_{t,c}} \\
P_t         &= \frac{R_t}{Q_t}
\end{aligned}
$$

Equivalently, the aggregate price is a sales-weighted average of SKU prices:

$$
P_{t,c,r,b}
= \frac{\sum_{s \in \mathcal{S}_{c,r,b}} P_{t,c,r,b,s}\,Q_{t,c,r,b,s}}
       {\sum_{s \in \mathcal{S}_{c,r,b}} Q_{t,c,r,b,s}}
$$

---

## 4. Market Share

### Brand Value Share

Brand $b$'s share of category revenue within retailer $r$:

$$
MS^{R}_{t,c,r,b}
= \frac{R_{t,c,r,b}}{R_{t,c,r}}
= \frac{R_{t,c,r,b}}{\sum_{b' \in \mathcal{B}_{c,r}} R_{t,c,r,b'}}
$$

### Brand Volume Share

Brand $b$'s share of category equivalized volume within retailer $r$:

$$
MS^{V}_{t,c,r,b}
= \frac{V_{t,c,r,b}}{V_{t,c,r}}
= \frac{\sum_{s \in \mathcal{S}_{c,r,b}} w_s Q_{t,c,r,b,s}}
       {\sum_{b' \in \mathcal{B}_{c,r}}\sum_{s \in \mathcal{S}_{c,r,b'}} w_s Q_{t,c,r,b',s}}
$$

### Retailer Value Share

Retailer $r$'s share of category revenue:

$$
MS^{R}_{t,c,r}
= \frac{R_{t,c,r}}{R_{t,c}}
= \frac{R_{t,c,r}}{\sum_{r' \in \mathcal{R}_c} R_{t,c,r'}}
$$

### Retailer Volume Share

Retailer $r$'s share of category equivalized volume:

$$
MS^{V}_{t,c,r}
= \frac{V_{t,c,r}}{V_{t,c}}
= \frac{V_{t,c,r}}{\sum_{r' \in \mathcal{R}_c} V_{t,c,r'}}
$$

By construction:

$$
\sum_{b \in \mathcal{B}_{c,r}} MS^{R}_{t,c,r,b} = 1
\qquad\text{and}\qquad
\sum_{r \in \mathcal{R}_c} MS^{R}_{t,c,r} = 1
$$

The same normalization applies to volume shares.

---

## 5. Growth Dynamics

### Month-over-Month Growth

For any metric $X \in \{Q,R,V,P\}$ at any hierarchy node $h$:

$$
g^{\text{MoM}}_{t,h,X}
= \frac{X_{t,h}-X_{t-1,h}}{X_{t-1,h}}
= \frac{X_{t,h}}{X_{t-1,h}} - 1
$$

**Example** (brand revenue growth):

$$
g^{\text{MoM}}_{t,c,r,b,R}
= \frac{R_{t,c,r,b}}{R_{t-1,c,r,b}} - 1
$$

### Year-over-Year Growth

With 24 monthly periods, YoY growth is available for $t = 13,\dots,24$:

$$
g^{\text{YoY}}_{t,h,X}
= \frac{X_{t,h}-X_{t-12,h}}{X_{t-12,h}}
= \frac{X_{t,h}}{X_{t-12,h}} - 1
$$

**Example** (brand equivalized-volume growth):

$$
g^{\text{YoY}}_{t,c,r,b,V}
= \frac{V_{t,c,r,b}}{V_{t-12,c,r,b}} - 1
$$

> YoY growth is preferred for performance assessment because it compares the same calendar month and therefore controls for regular seasonality.

### Log Growth

For additive decompositions and regression modeling:

$$
\Delta \ln X_{t,h}
= \ln X_{t,h} - \ln X_{t-1,h}
$$

For small changes:

$$
\Delta \ln X_{t,h} \approx g^{\text{MoM}}_{t,h,X}
$$

---

## 6. Share and Growth Relationship

Brand revenue growth can be decomposed into category growth and brand share momentum:

$$
\frac{R_{t,c,r,b}}{R_{t-1,c,r,b}}
= \frac{R_{t,c,r}}{R_{t-1,c,r}}
\times \frac{MS^{R}_{t,c,r,b}}{MS^{R}_{t-1,c,r,b}}
$$

Taking logarithms:

$$
\Delta \ln R_{t,c,r,b}
= \Delta \ln R_{t,c,r}
+ \Delta \ln MS^{R}_{t,c,r,b}
$$

Therefore:

$$
g^{\text{MoM}}_{t,c,r,b,R}
\approx g^{\text{MoM}}_{t,c,r,R}
+ \Delta \ln MS^{R}_{t,c,r,b}
$$

A brand can grow despite category decline if its share gain exceeds the category decline.

---

## 7. Own-Price Elasticity

### Theoretical Definition

For SKU $s$, own-price elasticity of demand is:

$$
\varepsilon_{s,s}
= \frac{\partial \ln Q_{t,c,r,b,s}}{\partial \ln P_{t,c,r,b,s}}
$$

**Expected sign:** $\varepsilon_{s,s} < 0$

### Baseline Log-Log Demand Model

$$
\begin{aligned}
\ln Q_{t,c,r,b,s}
=&\; \alpha_s
+ \beta_s \ln P_{t,c,r,b,s} \\
&+ \sum_{m=2}^{12} \lambda_{s,m} \text{Season}_{t,m} \\
&+ \tau_s \text{Trend}_t
+ \mu_r
+ u_{t,c,r,b,s}
\end{aligned}
$$

Where:

| Symbol | Meaning |
|--------|---------|
| $\alpha_s$ | SKU-specific baseline demand |
| $\beta_s$ | own-price elasticity |
| $\text{Season}_{t,m}$ | monthly seasonal dummy ($m=2,\dots,12$) |
| $\lambda_{s,m}$ | seasonal effect for month $m$ |
| $\text{Trend}_t = t$ | linear time trend |
| $\tau_s$ | monthly trend coefficient |
| $\mu_r$ | retailer fixed effect |
| $u_{t,c,r,b,s}$ | residual demand shock |

The estimated own-price elasticity is:

$$
\widehat{\varepsilon}_{s,s} = \widehat{\beta}_s
$$

**Interpretation:**

> A 1% increase in $P_s$ is associated with an estimated $\beta_s\%$ change in $Q_s$.

For a normal downward-sloping demand curve, $\beta_s < 0$. In a log-log regression, the coefficient on log price directly estimates elasticity.

### Arc Elasticity

For a two-period calculation:

$$
\varepsilon^{\text{arc}}_{s,s}
= \frac{\Delta Q_{t,c,r,b,s} / \bar{Q}_{t,c,r,b,s}}
       {\Delta P_{t,c,r,b,s} / \bar{P}_{t,c,r,b,s}}
$$

where:

$$
\bar{Q}_{t,c,r,b,s} = \frac{Q_{t,c,r,b,s} + Q_{t-1,c,r,b,s}}{2}
\qquad
\bar{P}_{t,c,r,b,s} = \frac{P_{t,c,r,b,s} + P_{t-1,c,r,b,s}}{2}
$$

---

## 8. Cross-Price Elasticity

For two SKUs $s$ and $j$ in the same category:

$$
\varepsilon_{s,j}
= \frac{\partial \ln Q_{t,c,r,b,s}}{\partial \ln P_{t,c,r,b,j}},
\qquad s \neq j
$$

### Pairwise Model

$$
\begin{aligned}
\ln Q_{t,c,r,b,s}
=&\; \alpha_s
+ \beta_s \ln P_{t,c,r,b,s}
+ \gamma_{s,j} \ln P_{t,c,r,b,j} \\
&+ \sum_{m=2}^{12} \lambda_{s,m} \text{Season}_{t,m} \\
&+ \tau_s \text{Trend}_t
+ \mu_r
+ u_{t,c,r,b,s}
\end{aligned}
$$

The estimated cross-price elasticity is:

$$
\widehat{\varepsilon}_{s,j} = \widehat{\gamma}_{s,j}
$$

### Interpretation

$$
\begin{aligned}
\varepsilon_{s,j} > 0 &\implies \text{SKUs } s \text{ and } j \text{ are substitutes} \\
\varepsilon_{s,j} < 0 &\implies \text{SKUs } s \text{ and } j \text{ are complements} \\
\varepsilon_{s,j} \approx 0 &\implies \text{No measurable substitution or complementarity}
\end{aligned}
$$

### Cannibalization

If $s$ and $j$ belong to the same brand:

$$
s,j \in \mathcal{S}_{c,r,b}
\quad\text{and}\quad
\varepsilon_{s,j} > 0
$$

then SKU $j$'s price increase is associated with higher demand for SKU $s$. Equivalently, a discount on $j$ may shift demand away from $s$, indicating potential cannibalization.

The approximate volume transferred is:

$$
\Delta Q^{\text{Cannibalized}}_{s \leftarrow j}
= \varepsilon_{s,j}
\times \frac{\Delta P_{j}}{P_{j}}
\times Q_s
$$

### Competitor Price Index

To avoid estimating every SKU pair, construct a competitor price index for SKU $s$:

$$
P^{\text{Comp}}_{t,c,r,-s}
= \sum_{j \neq s} \omega_{t,c,r,j}\, P_{t,c,r,b,j}
$$

with equivalized-volume weights:

$$
\omega_{t,c,r,j}
= \frac{V_{t,c,r,b,j}}{\sum_{j' \neq s} V_{t,c,r,b,j'}}
$$

Then estimate:

$$
\begin{aligned}
\ln Q_{t,c,r,b,s}
=&\; \alpha_s
+ \beta_s \ln P_{t,c,r,b,s}
+ \gamma_s \ln P^{\text{Comp}}_{t,c,r,-s} \\
&+ \sum_{m=2}^{12} \lambda_{s,m} \text{Season}_{t,m} \\
&+ \tau_s \text{Trend}_t
+ \mu_r
+ u_{t,c,r,b,s}
\end{aligned}
$$

**Expected sign:** $\gamma_s > 0$

If competitor prices rise, demand for SKU $s$ should increase through substitution.

---

## 9. Price-Volume Decomposition

Let Year 1 be period $0$ and Year 2 be period $1$. For any hierarchy node $h$:

$$
\begin{aligned}
R_{0,h} &= P_{0,h} Q_{0,h} \\
R_{1,h} &= P_{1,h} Q_{1,h} \\
\Delta R_h &= R_{1,h} - R_{0,h}
\end{aligned}
$$

Using current-period volume for the price effect:

$$
\begin{aligned}
PE_h &= (P_{1,h} - P_{0,h})\, Q_{1,h} \\
VE_h &= (Q_{1,h} - Q_{0,h})\, P_{0,h}
\end{aligned}
$$

Then:

$$
\Delta R_h = PE_h + VE_h
$$

### Percentage Contribution

$$
\begin{aligned}
\text{Price contribution}  &= \frac{PE_h}{R_{0,h}} \\
\text{Volume contribution} &= \frac{VE_h}{R_{0,h}} \\
g^{\text{YoY}}_{h,R} &= \frac{\Delta R_h}{R_{0,h}}
= \text{Price contribution} + \text{Volume contribution}
\end{aligned}
$$

> This is the standard two-factor decomposition: price effect equals the price change multiplied by current volume, while volume effect equals the volume change multiplied by base-period price.

---

## 10. Multi-SKU Mix Effect

When aggregating across SKUs, revenue changes may also reflect a shift in portfolio composition. Define:

$$
\begin{aligned}
PE_{c,r,b}
&= \sum_{s \in \mathcal{S}_{c,r,b}}
   (P_{1,c,r,b,s} - P_{0,c,r,b,s})\, Q_{1,c,r,b,s} \\
VE_{c,r,b}
&= \sum_{s \in \mathcal{S}_{c,r,b}}
   (Q_{1,c,r,b,s} - Q_{0,c,r,b,s})\, P_{0,c,r,b,s}
\end{aligned}
$$

The mix effect is the residual:

$$
ME_{c,r,b}
= \Delta R_{c,r,b} - PE_{c,r,b} - VE_{c,r,b}
$$

Thus:

$$
\Delta R_{c,r,b}
= PE_{c,r,b} + VE_{c,r,b} + ME_{c,r,b}
$$

A positive mix effect means revenue increased because the brand or category shifted toward higher-priced SKUs or pack formats.

---

## 11. Seasonality

Define monthly indicators:

$$
\text{Season}_{t,m}
= \begin{cases}
1, & \text{if month}(t) = m \\
0, & \text{otherwise}
\end{cases}
$$

With January as the reference month:

$$
\ln Q_{t,c,r,b,s}
= \alpha_s
+ \sum_{m=2}^{12} \lambda_{s,m} \text{Season}_{t,m}
+ u_{t,c,r,b,s}
$$

The multiplicative seasonal index is:

$$
SF_{s,m} = e^{\lambda_{s,m}}
$$

**Interpretation:**

$$
\begin{aligned}
SF_{s,m} > 1 &\implies \text{above-baseline seasonal demand} \\
SF_{s,m} < 1 &\implies \text{below-baseline seasonal demand}
\end{aligned}
$$

Seasonally adjusted unit sales are:

$$
Q^{\text{SA}}_{t,c,r,b,s}
= \frac{Q_{t,c,r,b,s}}{SF_{s,\text{month}(t)}}
$$

> With only two annual cycles, seasonal estimates should be treated as indicative rather than definitive.

---

## 12. Effect Map

| Effect | Equation | Expected Sign | Identifiable from POS-Only Data |
|--------|----------|:-------------:|:-------------------------------:|
| Own price | $\varepsilon_{s,s} = \partial \ln Q_s / \partial \ln P_s$ | Negative | Yes, with caution |
| Cross-price | $\varepsilon_{s,j} = \partial \ln Q_s / \partial \ln P_j$ | +: substitute; $-$: complement | Yes, with caution |
| Cannibalization | $\varepsilon_{s,j} > 0$ within same brand/category | Volume transfer | Partially |
| Seasonality | $SF_{s,m} = e^{\lambda_{s,m}}$ | Varies by month | Partially; two annual cycles |
| Trend | $\ln Q_t = \alpha + \tau t$ | + or $-$ | Yes |
| Pack-size effect | $P^V = P / w_s$ | Usually $-$ for larger packs | Yes |
| Mix | $ME = \Delta R - PE - VE$ | + or $-$ | Yes |
| Market share | $MS^R = R_b / R_{\text{category}}$ | + or $-$ | Yes |
| Retailer effect | $\mu_r$ | + or $-$ | Yes |
| Residual shock | $u_{t,c,r,b,s}$ | + or $-$ | Yes, but unexplained |
| Promotion | — | Usually + | **No** |
| Distribution | — | + | **No** |
| Advertising | — | Usually + | **No** |

---

## 13. Recommended Reporting

For each brand, retailer, and category, report:

1. Revenue, unit sales, and equivalized volume
2. Revenue-weighted price per unit and price per standardized volume
3. Value share and volume share
4. MoM and YoY growth
5. Price effect and volume effect
6. Mix effect for multi-SKU brands and categories
7. Own-price elasticity by SKU or brand
8. Cross-price elasticities for key substitute pairs
9. Seasonal indices by month
10. Residual demand shocks for further investigation

> **Caution:** Elasticity estimates from observational POS data are **associations under observed retail conditions**, not guaranteed causal effects. Unobserved promotions, stock changes, and local demand shocks may coincide with price movements.
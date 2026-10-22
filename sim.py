import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

# 1. Load Data
df = pd.read_csv('sample_data.csv')

# Preprocessing for log-log model
df['log_Q'] = np.log(df['Unit_Sales'])
df['log_P'] = np.log(df['Price'])

# Create categorical encodings for PyMC
sku_idxs, skus = pd.factorize(df['SKU'])
ret_idxs, retailers = pd.factorize(df['Retailer'])
month_idxs = df['Month'].values - 1  # 0 to 23

coords = {
    'sku': skus,
    'retailer': retailers,
    'obs_id': np.arange(len(df))
}

# 2. Build the Bayesian Demand Model
with pm.Model(coords=coords) as retail_model:
    # Data containers (allows us to swap data later for simulation)
    sku_data = pm.Data('sku_data', sku_idxs)
    ret_data = pm.Data('ret_data', ret_idxs)
    month_data = pm.Data('month_data', month_idxs)
    log_P_data = pm.Data('log_P_data', df['log_P'].values)
    
    # Priors
    # Alpha (Base demand per SKU)
    alpha = pm.Normal('alpha', mu=8, sigma=2, dims='sku')
    
    # Beta (Own-price elasticity per SKU) 
    # We expect elasticity to be negative, so we center it around -1.5
    beta = pm.Normal('beta', mu=-1.5, sigma=1, dims='sku')
    
    # Retailer fixed effects (centered at 0)
    mu_ret = pm.Normal('mu_ret', mu=0, sigma=0.5, dims='retailer')
    
    # Seasonal effect (using a simple random walk or independent normals)
    # We use month_data % 12 to capture the repeating 12-month cycle
    month_of_year = month_data % 12
    lambda_m = pm.Normal('lambda_m', mu=0, sigma=0.3, shape=12)
    
    # Expected log units
    mu = (alpha[sku_data] + 
          beta[sku_data] * log_P_data + 
          mu_ret[ret_data] + 
          lambda_m[month_of_year])
    
    # Error variance
    sigma = pm.Exponential('sigma', 1)
    
    # Likelihood
    log_Q_obs = pm.Normal('log_Q_obs', mu=mu, sigma=sigma, observed=df['log_Q'].values)
    
    # 3. Sample the posterior
    print("Fitting model to historical data...")
    trace = pm.sample(draws=1000, tune=1000, target_accept=0.9, cores=2)

# 4. Prepare Counterfactual Scenario
# What if 'Nexus_Retail_Group' slashed the price of 'Cola_2L_Bottle' by 20% across all months?
df_sim = df.copy()

mask = (df_sim['Retailer'] == 'Nexus_Retail_Group') & (df_sim['SKU'] == 'Cola_2L_Bottle')
df_sim.loc[mask, 'Price'] = df_sim.loc[mask, 'Price'] * 0.80  # 20% discount
df_sim['log_P'] = np.log(df_sim['Price'])

# 5. Run the Simulation
with retail_model:
    # Update the data containers with the new simulated dataset
    pm.set_data({
        'log_P_data': df_sim['log_P'].values,
        'sku_data': sku_idxs,
        'ret_data': ret_idxs,
        'month_data': month_idxs
    })
    
    print("Simulating counterfactual prices...")
    # Sample from the posterior predictive distribution using the new prices
    post_pred = pm.sample_posterior_predictive(trace, var_names=['log_Q_obs'])

# 6. Analyze the Results
# Extract the simulated log_Q arrays (samples x observations)
simulated_log_Q = post_pred.posterior_predictive['log_Q_obs'].values
# Average across the chains and samples to get the mean expected log_Q
mean_simulated_log_Q = simulated_log_Q.mean(axis=(0, 1))

# Convert back to actual units and calculate revenue
df_sim['Simulated_Unit_Sales'] = np.exp(mean_simulated_log_Q).astype(int)
df_sim['Simulated_Revenue'] = df_sim['Simulated_Unit_Sales'] * df_sim['Price']

# Compare the Baseline vs Simulated Scenario for the targeted SKU/Retailer
baseline_revenue = df.loc[mask, 'Sales_Revenue'].sum()
simulated_revenue = df_sim.loc[mask, 'Simulated_Revenue'].sum()

print("\n--- SIMULATION RESULTS: 20% Price Cut on Cola_2L_Bottle at Nexus_Retail_Group ---")
print(f"Historical Revenue: ${baseline_revenue:,.2f}")
print(f"Simulated Revenue:  ${simulated_revenue:,.2f}")
print(f"Revenue Delta:      ${simulated_revenue - baseline_revenue:,.2f}")

# Plotting the before/after for visualization
nexus_cola_baseline = df.loc[mask].groupby('Month')['Sales_Revenue'].sum()
nexus_cola_simulated = df_sim.loc[mask].groupby('Month')['Simulated_Revenue'].sum()

plt.figure(figsize=(10, 5))
plt.plot(nexus_cola_baseline.index, nexus_cola_baseline.values, label='Historical Revenue', marker='o')
plt.plot(nexus_cola_simulated.index, nexus_cola_simulated.values, label='Simulated Revenue (-20% Price)', marker='x', linestyle='--')
plt.title('Counterfactual Simulation: Cola 2L Bottle at Nexus Retail Group')
plt.xlabel('Month')
plt.ylabel('Revenue ($)')
plt.legend()
plt.grid(True)
plt.show()
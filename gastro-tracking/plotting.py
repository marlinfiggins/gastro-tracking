import matplotlib.pyplot as plt
import numpy as np
from matplotlib.dates import DateFormatter


def plot_symptom_presence_over_time(data, symptoms, figsize=None):
    """
    Plot symptoms over time using a grid layout.

    Args:
    - data (pd.DataFrame): The simulated data containing symptoms and date information.
    - symptoms (list): A list of symptom names to plot.
    - figsize (tuple): Size of the figure (optional).
    """
    # Group by date and compute presence/absence for each symptom
    grouped_data = data.groupby('Date')[symptoms].any().astype(int)

    # Create a figure with GridSpec layout
    fig = plt.figure(figsize=figsize if figsize is not None else (16, 8))
    grid = fig.add_gridspec(len(symptoms), 1, figure=fig, hspace=0.4)

    date_formatter = DateFormatter("%b %d")
    for i, symptom in enumerate(symptoms):
        ax = fig.add_subplot(grid[i, 0])
        ax.bar(grouped_data.index, grouped_data[symptom], ec="k")
        ax.xaxis.set_major_formatter(date_formatter)
        ax.set_title(symptom)
        ax.set_ylabel("Presence")
        ax.set_yticks([0,1])
        ax.grid(axis="x")
    return None

def plot_food_consumption_over_time(data, foods, figsize=None):
    """
    Plot food consumption over time using a grid layout.

    Args:
    - data (pd.DataFrame): The simulated data containing food and date information.
    - foods (list): A list of food names to plot.
    - figsize (tuple): Size of the figure (optional).
    """
    # Group by date and compute presence/absence for each food
    grouped_data = (
        data.assign(**{food: data["Foods"].str.contains(food) for food in foods})
        .groupby("Date")[foods]
        .any()
        .astype(int)
    )

    # Create a figure with GridSpec layout
    fig = plt.figure(figsize=figsize if figsize is not None else (16, 8))
    grid = fig.add_gridspec(len(foods), 1, figure=fig, hspace=0.4)

    date_formatter = DateFormatter("%b %d")
    for i, food in enumerate(foods):
        ax = fig.add_subplot(grid[i, 0])
        ax.bar(grouped_data.index, grouped_data[food], ec="k")
        ax.xaxis.set_major_formatter(date_formatter)
        ax.set_title(food)
        ax.set_ylabel("Presence")
        ax.set_yticks([0, 1])
        ax.grid(axis="x")
    return None

def plot_effects_with_intervals(mcmc_results, predictors, outcomes, figsize=None, show_legend=True):
    """
    Plot estimated effects as intervals for each outcome and predictor.

    Args:
    - mcmc_results: MCMC object containing posterior samples.
    - predictors (list): List of predictor names.
    - outcomes (list): List of outcome names.
    - figsize (tuple): Size of the figure (optional).
    - show_legend (bool): Show the legend (optional).
    """
    posterior_samples = mcmc_results.get_samples()
    num_outcomes = len(outcomes)
    num_predictors = len(predictors)
    fig, axes = plt.subplots(num_outcomes, 1, figsize=figsize if figsize is not None else (10,  4 * num_outcomes), sharex=True)

    if num_outcomes == 1:
        axes = [axes]  # Ensure axes is iterable for a single outcome

    for i, outcome in enumerate(outcomes):
        ax = axes[i]
        ax.set_title(f"Effects on {outcome}", fontsize=14)

        for j, predictor in enumerate(predictors):
            posterior_key = f"Coefficient_{outcome}_{predictor}"
            if posterior_key in posterior_samples:
                posterior_values = np.array(posterior_samples[posterior_key])

                # Compute HDIs using NumPy quantiles
                hdi_95 = np.quantile(posterior_values, [0.025, 0.975])
                hdi_80 = np.quantile(posterior_values, [0.1, 0.9])
                hdi_50 = np.quantile(posterior_values, [0.25, 0.75])

                # Plot intervals
                ax.plot([j, j], hdi_95, color="C0", alpha=0.5, linewidth=6, label="95% HDI" if j == 0 else None)
                ax.plot([j, j], hdi_80, color="C0", alpha=0.7, linewidth=10, label="80% HDI" if j == 0 else None)
                ax.plot([j, j], hdi_50, color="C0", alpha=1.0, linewidth=14, label="50% HDI" if j == 0 else None)

                # Plot median
                median = np.median(posterior_values)
                ax.scatter(j, median, color="C1", zorder=5)

        # Formatting
        ax.axhline(0, color="k", linestyle="--", linewidth=1, alpha=0.8)
        ax.set_xticks(range(num_predictors))
        ax.set_xticklabels(predictors)
        ax.set_ylabel("Effect")
        if i == 0 and show_legend:
            ax.legend(loc="best", fontsize=10)
    return None

def plot_risks_with_predictors(mcmc_results, predictors, outcomes, baseline_X):
    """
    Plot the baseline and risk-modified probabilities for each outcome and predictor.

    Args:
    - mcmc_results: MCMC object containing posterior samples.
    - predictors (list): List of predictor names.
    - outcomes (list): List of outcome names.
    - baseline_X (jnp.array): Baseline predictor matrix (all predictors set to 0 for baseline risk).
    """
    posterior_samples = mcmc_results.get_samples()
    num_outcomes = len(outcomes)
    fig, axes = plt.subplots(num_outcomes, 1, figsize=(10, 6 * num_outcomes), sharex=True)

    if num_outcomes == 1:
        axes = [axes]  # Ensure axes is iterable for a single outcome

    for i, outcome in enumerate(outcomes):
        ax = axes[i]
        ax.set_title(f"Risk Effects on {outcome}", fontsize=14)

        baseline_logits = np.array(posterior_samples["Intercepts"][:, i])
        baseline_probs = np.exp(baseline_logits) / (1 + np.exp(baseline_logits))

        for j, predictor in enumerate(predictors):
            predictor_logits = baseline_logits + np.array(posterior_samples[f"Coefficient_{outcome}_{predictor}"])
            predictor_probs = np.exp(predictor_logits) / (1 + np.exp(predictor_logits))

            # Compute Risk Difference
            risk_difference = predictor_probs.mean(axis=0) - baseline_probs.mean(axis=0)

            # Plot risk changes
            ax.bar(j, risk_difference, color="C0", alpha=0.8)
            ax.text(j, risk_difference, f"{risk_difference:.2f}", ha="center", va="bottom", fontsize=10)

        # Formatting
        ax.axhline(0, color="k", linestyle="--", linewidth=1, alpha=0.8)
        ax.set_xticks(range(len(predictors)))
        ax.set_xticklabels(predictors)
        ax.set_ylabel("Risk Difference")
    return None

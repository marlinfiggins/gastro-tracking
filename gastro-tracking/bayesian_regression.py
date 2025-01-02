import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS


class BayesianRegression:
    def __init__(self, predictors, outcomes, default_priors=None):
        """
        Base class for Bayesian regression models.

        Args:
        - predictors (list): List of predictor names.
        - outcomes (list): List of outcome names.
        - default_priors (dict): Dictionary specifying priors for each predictor for each outcome.
                                 Format: {outcome: {predictor: prior_std}}
        """
        self.predictors = predictors
        self.outcomes = outcomes
        self.default_priors = default_priors if default_priors else {}

    def define_priors(self):
        """
        Define priors for intercepts and coefficients.

        Returns:
        - tuple: Intercepts and coefficients priors for outcomes and predictors.
        """
        intercepts = numpyro.sample("Intercepts", dist.Normal(0, 10).expand([len(self.outcomes)]))
        coefficients = {}

        for outcome in self.outcomes:
            coeffs = []
            for predictor in self.predictors:
                prior_std = self.default_priors.get(outcome, {}).get(predictor, 2)
                coeff = numpyro.sample(f"Coefficient_{outcome}_{predictor}", dist.Normal(0, prior_std))
                coeffs.append(coeff)
            coefficients[outcome] = jnp.array(coeffs)

        return intercepts, coefficients

    def fit(self, X, Y, num_samples=1000, num_warmup=500, num_chains=1):
        """
        Fit the model using MCMC.

        Args:
        - X (jnp.array): Predictor matrix (n_samples x n_predictors).
        - Y (dict): Dictionary of outcome arrays (n_samples) indexed by outcome names.
        - num_samples (int): Number of posterior samples to draw.
        - num_warmup (int): Number of warmup steps.
        - num_chains (int): Number of MCMC chains.

        Returns:
        - MCMC: MCMC object containing the posterior samples.
        """
        kernel = NUTS(self.model)
        mcmc = MCMC(kernel, num_samples=num_samples, num_warmup=num_warmup, num_chains=num_chains)
        mcmc.run(jax.random.PRNGKey(0), X, Y)
        mcmc.print_summary()
        self.mcmc = mcmc
        return mcmc

    def predict(self, X_new):
        """
        Dispatch prediction to the subclass-defined method.
        """
        raise NotImplementedError("Subclasses must implement this method.")


class BayesianLogisticRegression(BayesianRegression):
    def model(self, X, Y):
        """
        Define the logistic regression model.
        """
        intercepts, coefficients = self.define_priors()
        for i, outcome in enumerate(self.outcomes):
            logits = intercepts[i] + jnp.dot(X, coefficients[outcome])
            numpyro.sample(outcome, dist.Bernoulli(logits=logits), obs=Y[outcome])

    def predict(self, X_new):
        """
        Predict probabilities for new data using posterior samples.
        """
        posterior_samples = self.mcmc.get_samples()
        intercepts = posterior_samples["Intercepts"]
        predictions = {}
    
        for i, outcome in enumerate(self.outcomes):
            coeffs = jnp.array([posterior_samples[f"Coefficient_{outcome}_{p}"] for p in self.predictors]).T
            logits = intercepts[:, i][:, None] + jnp.dot(coeffs, X_new.T)
            probabilities = jnp.exp(logits) / (1 + jnp.exp(logits))
            predictions[outcome] = probabilities.mean(axis=0)
    
        return predictions


class BayesianLinearRegression(BayesianRegression):
    def model(self, X, Y):
        """
        Define the linear regression model.
        """
        intercepts, coefficients = self.define_priors()
        for i, outcome in enumerate(self.outcomes):
            mean = intercepts[i] + jnp.dot(X, coefficients[outcome])
            numpyro.sample(outcome, dist.Normal(mean, 1), obs=Y[outcome])

    def predict(self, X_new):
        """
        Predict continuous outcomes using posterior samples.
        """
        posterior_samples = self.mcmc.get_samples()
        intercepts = posterior_samples["Intercepts"]
        predictions = {}

        for i, outcome in enumerate(self.outcomes):
            coeffs = jnp.array([posterior_samples[f"Coefficient_{outcome}_{p}"] for p in self.predictors]).T
            predictions[outcome] = (intercepts[:, i][:, None] + jnp.dot(coeffs, X_new.T).T).mean(axis=0)

        return predictions


class BayesianCountRegression(BayesianRegression):
    def model(self, X, Y):
        """
        Define the count regression model (Poisson likelihood).
        """
        intercepts, coefficients = self.define_priors()
        for i, outcome in enumerate(self.outcomes):
            logits = intercepts[i] + jnp.dot(X, coefficients[outcome])
            numpyro.sample(outcome, dist.Poisson(rate=jnp.exp(logits)), obs=Y[outcome])

    def predict(self, X_new):
        """
        Predict counts using posterior samples.
        """
        posterior_samples = self.mcmc.get_samples()
        intercepts = posterior_samples["Intercepts"]
        predictions = {}

        for i, outcome in enumerate(self.outcomes):
            coeffs = jnp.array([posterior_samples[f"Coefficient_{outcome}_{p}"] for p in self.predictors]).T
            logits = intercepts[:, i][:, None] + jnp.dot(coeffs, X_new.T).T
            predictions[outcome] = jnp.exp(logits).mean(axis=0)

        return predictions

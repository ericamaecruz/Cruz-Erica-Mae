# PROBLEM 1
# Coffee Consumption among Call Center Agents
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

bold_start = "\033[1m"
bold_end = "\033[0m"
print(f"{bold_start}Problem 1:{bold_end}\nThis code simulates a Bayesian model to estimate the average number of cups of coffee consumed daily by call center agents. We start with an initial belief that agents drink 3 cups per day on average and update it based on data collected from 50 agents.")

prior_mu = 3
prior_precision = 1.5
prior_sigma_alpha = 2
prior_sigma_beta = 1.5

np.random.seed(42)
true_mu = 3.8
true_sigma = 1.2

n = 50
data = np.random.normal(loc=true_mu, scale=true_sigma, size=n)

posterior_precision = prior_precision + n / true_sigma**2
posterior_mu = (prior_precision * prior_mu + np.sum(data) / true_sigma**2) / posterior_precision
posterior_sigma_alpha = prior_sigma_alpha + n / 2
posterior_sigma_beta = prior_sigma_beta + np.sum((data - np.mean(data)) ** 2) / 2

posterior_mu_samples = np.random.normal(posterior_mu, 1 / np.sqrt(posterior_precision), size=10000)
posterior_sigma_samples = np.sqrt(np.random.gamma(posterior_sigma_alpha, 1 / posterior_sigma_beta, size=10000))

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.hist(posterior_mu_samples, bins=30, density=True, color='sandybrown', edgecolor='black', alpha=0.7)
plt.title(r"Posterior Distribution of Mean Cups of Coffee ($\mu$)")
plt.xlabel('Mean Cups of Coffee')
plt.ylabel('Density')

plt.subplot(1, 2, 2)
plt.hist(posterior_sigma_samples, bins=30, density=True, color='sandybrown', edgecolor='black', alpha=0.7)
plt.title(r"Posterior Distribution of Standard Deviation ($\sigma$)")
plt.xlabel('Standard Deviation')
plt.ylabel('Density')

plt.tight_layout()
plt.show()

print("Estimated mean cups of coffee consumed daily:", np.mean(posterior_mu_samples))
print("Standard deviation of the estimated mean:", np.std(posterior_mu_samples))
print("Estimated standard deviation in daily coffee consumption:", np.mean(posterior_sigma_samples))
print("Standard deviation of the estimated standard deviation:", np.std(posterior_sigma_samples))


# PROBLEM 2
# Calorie Intake among Gym Rats
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

print(f"\n\n{bold_start}Problem 2:{bold_end}\nThis code is a simulation of a Bayesian model to estimate how many calories gym rats consume on average each day. We begin with an initial assumption that the average intake is 2800 calories per day, and then update this assumption based on data gathered from 80 gym-goers.")

prior_mu = 2800
prior_precision = 1e-4
prior_sigma_alpha = 3
prior_sigma_beta = 100000

np.random.seed(42)
true_mu = 3200
true_sigma = 400

n = 80
data = np.random.normal(loc=true_mu, scale=true_sigma, size=n)

posterior_precision = prior_precision + n / true_sigma**2
posterior_mu = (prior_precision * prior_mu + np.sum(data) / true_sigma**2) / posterior_precision
posterior_sigma_alpha = prior_sigma_alpha + n / 2
posterior_sigma_beta = prior_sigma_beta + np.sum((data - np.mean(data)) ** 2) / 2

posterior_mu_samples = np.random.normal(posterior_mu, 1 / np.sqrt(posterior_precision), size=10000)
posterior_sigma_samples = np.sqrt(np.random.gamma(posterior_sigma_alpha, 1 / posterior_sigma_beta, size=10000))

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.hist(posterior_mu_samples, bins=30, density=True, color='lightcoral', edgecolor='black', alpha=0.7)
plt.title(r"Posterior Distribution of Mean Calorie Intake ($\mu$)")
plt.xlabel('Mean Calories')
plt.ylabel('Density')

plt.subplot(1, 2, 2)
plt.hist(posterior_sigma_samples, bins=30, density=True, color='lightcoral', edgecolor='black', alpha=0.7)
plt.title(r"Posterior Distribution of Standard Deviation ($\sigma$)")
plt.xlabel('Standard Deviation (Calories)')
plt.ylabel('Density')

plt.tight_layout()
plt.show()

print("Estimated mean daily calorie intake among gym rats:", np.mean(posterior_mu_samples))
print("Standard deviation of the estimated mean:", np.std(posterior_mu_samples))
print("Estimated standard deviation in daily intake:", np.mean(posterior_sigma_samples))
print("Standard deviation of the estimated standard deviation:", np.std(posterior_sigma_samples))


# PROBLEM 3
# Scores per Game of a Basketball Player
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

print(f"\n\n{bold_start}Problem 3:{bold_end}\nThis code simulates a Bayesian model to estimate the average number of points a basketball player scores per game. We start with an initial belief of 18 points per game and refine it based on 40 observed games.")


prior_mu = 18
prior_precision = 0.05
prior_sigma_alpha = 2
prior_sigma_beta = 30

np.random.seed(42)
true_mu = 22
true_sigma = 5

n = 40
data = np.random.normal(loc=true_mu, scale=true_sigma, size=n)

posterior_precision = prior_precision + n / true_sigma**2
posterior_mu = (prior_precision * prior_mu + np.sum(data) / true_sigma**2) / posterior_precision
posterior_sigma_alpha = prior_sigma_alpha + n / 2
posterior_sigma_beta = prior_sigma_beta + np.sum((data - np.mean(data)) ** 2) / 2

posterior_mu_samples = np.random.normal(posterior_mu, 1 / np.sqrt(posterior_precision), size=10000)
posterior_sigma_samples = np.sqrt(np.random.gamma(posterior_sigma_alpha, 1 / posterior_sigma_beta, size=10000))

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.hist(posterior_mu_samples, bins=30, density=True, color='orange', edgecolor='black', alpha=0.7)
plt.title(r"Posterior Distribution of Mean Points per Game ($\mu$)")
plt.xlabel('Points per Game')
plt.ylabel('Density')

plt.subplot(1, 2, 2)
plt.hist(posterior_sigma_samples, bins=30, density=True, color='orange', edgecolor='black', alpha=0.7)
plt.title(r"Posterior Distribution of Standard Deviation ($\sigma$)")
plt.xlabel('Standard Deviation')
plt.ylabel('Density')

plt.tight_layout()
plt.show()

print("Estimated average points per game:", np.mean(posterior_mu_samples))
print("Standard deviation of the estimated mean:", np.std(posterior_mu_samples))
print("Estimated variability in points per game:", np.mean(posterior_sigma_samples))
print("Standard deviation of the estimated variability:", np.std(posterior_sigma_samples))
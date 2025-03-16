import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
true_mu = 12
true_sigma = 11
data = np.random.normal(true_mu, true_sigma, size = 150)

prior_mu_mean = 2
prior_mu_precision = 4
prior_sigma_alpha = 2
prior_sigma_beta = 3

posterior_mu_precision = prior_mu_precision + len(data)/true_sigma**2
posterior_mu_mean = (prior_mu_precision*prior_mu_mean+np.sum(data))/posterior_mu_precision

posterior_sigma_alpha = prior_sigma_alpha + len(data)/2
posterior_sigma_beta = prior_sigma_beta + np.sum((data - np.mean(data))**2)/2

posterior_mu = np.random.normal(posterior_mu_mean, 1 / np.sqrt(posterior_mu_precision), size = 10000)
posterior_sigma = np.random.gamma(posterior_sigma_alpha, 1 / posterior_sigma_beta, size = 10000)

plt.figure(figsize=(10,4))
plt.subplot(1, 2, 1)
plt.hist(posterior_mu, bins = 30, density = True, color = 'yellow', edgecolor = 'black')
plt.title(r"Posterior distribution of $\mu$")
plt.xlabel(r"$\mu$")
plt.ylabel("Density")

plt.subplot(1, 2, 2)
plt.hist(posterior_sigma, bins = 30, density = True, color = 'red', edgecolor = 'black')
plt.title(r"Posterior distribution of $\sigma$")
plt.xlabel(r"$\sigma$")
plt.ylabel("Density")

plt.tight_layout()
plt.show()

mean_mu = np.mean(posterior_mu)
std_mu = np.std(posterior_mu)
print("Mean of mu:", mean_mu)
print("Standard Deviation of mu:", std_mu)

mean_sigma = np.mean(posterior_sigma)
std_sigma = np.std(posterior_sigma)
print("Mean of sigma:", mean_sigma)
print("Standard Deviation of sigma:", std_sigma)
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def truncated_normal_pdf(x, mean, std, low, high):
    # Calculate the PDF of the normal distribution
    pdf = np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))

    # Truncate the distribution
    pdf[(x < low) | (x > high)] = 0

    return pdf


def sample_truncated_normal_integers(low, high, mean, std, size):
    # Generate the range of integers
    x = np.arange(low, high + 1)

    # Calculate the PDF values for the range of integers
    probabilities = truncated_normal_pdf(x, mean, std, low, high)

    # Normalize the probabilities
    probabilities /= probabilities.sum()

    # Sample integers based on the normalized probabilities
    samples = np.random.choice(x, size=size, p=probabilities)

    return samples


# %% Example usage integers
# low = 5
# high = 20
# mean = 15
# std = 8
# size = 1000

# low = 20
# high = 100
# mean = 60
# std = 60
# size = 5000

low = 10
high = 60
mean = 35
std = 30
size = 3000


samples = sample_truncated_normal_integers(low, high, mean, std, size)
print(samples)

plt.hist(samples, bins=range(low, high + 2))
plt.show()

# %% Float version


def sample_truncated_normal_floats(low, high, mean, std, size):
    # Generate a large number of samples from the normal distribution
    samples = np.random.normal(loc=mean, scale=std, size=size * 10)

    # Truncate the samples
    samples = samples[(samples >= low) & (samples <= high)]

    # If we don't have enough samples, repeat the process
    while len(samples) < size:
        additional_samples = np.random.normal(loc=mean, scale=std, size=size * 10)
        additional_samples = additional_samples[
            (additional_samples >= low) & (additional_samples <= high)
        ]
        samples = np.concatenate((samples, additional_samples))

    # Select the required number of samples
    samples = samples[:size]

    return samples


# %% Example usage floats
low = 2
high = 8
mean = 5
std = 3
size = 5000

samples = sample_truncated_normal_floats(low, high, mean, std, size)
print(samples)

plt.hist(samples, bins=25, color="r")
plt.show()
# %%

# # Use weibull for the spacing, #! DON'T USE THIS THE TAIL END IS TOO LONG
# weibull = stats.weibull_min(2, loc=2, scale=3)
# samples = weibull.rvs(size=5000)
# plt.hist(samples, bins=50)

# %% Example usage floats
low = 10
high = 25
mean = 15
std = 8
size = 5000

samples = sample_truncated_normal_floats(low, high, mean, std, size)
print(samples)

plt.hist(samples, bins=25, color="r")
plt.show()
# %%

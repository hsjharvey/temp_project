import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, skew, kurtosis

# --- Generate synthetic BTC-AUD data ---
np.random.seed(42)
btc_aud = pd.read_csv('BTC-AUD.csv')
btc_aud = btc_aud["daily_return"]
btc_aud = btc_aud
asx_aud = pd.read_csv('ASX200.csv')
asx_aud = asx_aud["daily_return"]

print(btc_aud.shape)
print(asx_aud.shape)

# --- Compute central moments ---
mean_btc = np.mean(btc_aud)
var_btc = np.var(btc_aud)
skew_btc = skew(btc_aud)
kurt_btc = kurtosis(btc_aud)

mean_asx = np.mean(asx_aud)
var_asx = np.var(asx_aud)
skew_asx = skew(asx_aud)
kurt_asx = kurtosis(asx_aud)

# --- Create figure with two subplots ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor='none')

# ---------------- Left Plot: Histogram + PDF ----------------
ax1 = axes[0]

# Background histogram
ax1.hist(asx_aud, bins=200, color='orange', alpha=0.5, label='ASX200')

# Foreground histogram
ax1.hist(btc_aud[:1500], bins=200, color='skyblue', alpha=0.5, label='BTC-AUD')

# Title, labels, legend
ax1.set_title('Histogram', fontsize=16)
ax1.set_xlabel('Daily return', fontsize=16)
ax1.set_ylabel('Frequency', fontsize=16)
ax1.legend(fontsize=16)

# Annotation box with central moments
textstr = (f"BTC-AUD:\nMean={mean_btc:.4f}\nVar={var_btc:.4f}\nSkew={skew_btc:.2f}\nKurt={kurt_btc:.2f}\n\n")
ax1.text(0.95, 0.95, textstr, transform=ax1.transAxes,
         fontsize=14, verticalalignment='top', horizontalalignment='right')

textstr = (f"ASX200:\nMean={mean_asx:.4f}\nVar={var_asx:.4f}\nSkew={skew_asx:.2f}\nKurt={kurt_asx:.2f}\n\n")
ax1.text(0.3, 0.3, textstr, transform=ax1.transAxes,
         fontsize=14, verticalalignment='top', horizontalalignment='right')

# ---------------- Right Plot: CDF Comparison ----------------
ax2 = axes[1]

# Empirical CDF for BTC-AUD
sorted_data = np.sort(asx_aud)
cdf_btc = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
ax2.plot(sorted_data, cdf_btc, '-', color="orange", lw=2, label='ASX200')

sorted_data = np.sort(btc_aud)
cdf_btc = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
ax2.plot(sorted_data, cdf_btc, '-', color="skyblue", lw=2, label='BTC-AUD')

# Title, labels, legend
ax2.set_title('CDF', fontsize=16)
ax2.set_xlabel('Daily return', fontsize=16)
ax2.set_ylabel('Cumulative Probability', fontsize=16)
ax2.legend(fontsize=16)

# Adjust tick font sizes
for ax in axes:
    ax.tick_params(axis='both', which='major', labelsize=16)

plt.tight_layout()
plt.show()

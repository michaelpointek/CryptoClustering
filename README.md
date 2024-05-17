# Cryptocurrency Clustering with PCA and K-Means

This project aims to cluster cryptocurrencies based on their market data using K-Means clustering and Principal Component Analysis (PCA) for dimensionality reduction. The goal is to identify patterns and group similar cryptocurrencies together for better market analysis.

## Table of Contents

- [Overview](#overview)
- [Data](#data)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview

Cryptocurrencies exhibit varied behaviors and characteristics in the market. By clustering them, we can identify groups of similar cryptocurrencies, which can help in making investment decisions and understanding market dynamics. This project uses K-Means clustering to group cryptocurrencies and PCA to reduce the dimensionality of the dataset for easier visualization and analysis.

## Data

The data used in this project includes various market metrics for cryptocurrencies, such as:

- Price change percentage in the last 24 hours
- Price change percentage in the last 7 days
- Market capitalization
- Trading volume

The dataset is stored in a CSV file named `crypto_market_data.csv`.

## Installation

To run this project, you need to have Python and the following libraries installed:

- pandas
- hvplot
- scikit-learn

You can install these dependencies using pip:

```bash
pip install pandas hvplot scikit-learn
```

## Usage

1. Clone the repository:

```bash
git clone https://github.com/yourusername/crypto-clustering.git
cd crypto-clustering
```

2. Load the data:

```python
import pandas as pd

df_market_data = pd.read_csv("Resources/crypto_market_data.csv", index_col="coin_id")
```

3. Normalize the data:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_market_data)
df_scaled = pd.DataFrame(scaled_data, columns=df_market_data.columns, index=df_market_data.index)
```

4. Compute inertia for different values of k:

```python
from sklearn.cluster import KMeans

k_values = list(range(1, 12))
inertia = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(df_scaled)
    inertia.append(kmeans.inertia_)

elbow_data = {"k": k_values, "inertia": inertia}
df_elbow = pd.DataFrame(elbow_data)
```

5. Plot the Elbow curve to determine the optimal number of clusters:

```python
import hvplot.pandas

elbow_plot = df_elbow.hvplot.line(x='k', y='inertia', title='Elbow Curve', xlabel='Number of Clusters (k)', ylabel='Inertia')
elbow_plot
```

6. Initialize and fit the K-Means model with the optimal number of clusters:

```python
best_k = 4
kmeans = KMeans(n_clusters=best_k, random_state=0)
kmeans.fit(df_scaled)
clusters = kmeans.predict(df_scaled)
```

7. Add the cluster labels to the DataFrame and visualize the clusters:

```python
df_scaled['cluster'] = clusters
scatter_plot = df_scaled.hvplot.scatter(x='price_change_percentage_24h', y='price_change_percentage_7d', by='cluster', hover_cols=['coin_id'], title='Cryptocurrency Clusters', xlabel='24h Price Change (%)', ylabel='7d Price Change (%)')
scatter_plot
```

8. Apply PCA for dimensionality reduction:

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=3)
pca_components = pca.fit_transform(df_scaled.drop(columns=['cluster']))
df_pca = pd.DataFrame(pca_components, columns=['PCA1', 'PCA2', 'PCA3'], index=df_scaled.index)
df_pca['cluster'] = clusters
df_pca.head()
```

## Results

The clustering results and PCA visualization help in understanding the grouping of cryptocurrencies. The Elbow curve method helps determine the optimal number of clusters for K-Means clustering.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the [Apache](LICENSE) file for details.

---

Feel free to customize this README to better fit your project's specifics.

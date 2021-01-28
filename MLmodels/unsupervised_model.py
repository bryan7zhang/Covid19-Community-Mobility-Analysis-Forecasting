import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import parallel_coordinates
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import dataselector

plt.rcParams['figure.figsize'] = [7, 6.4]
scaler = None


def process_features(x):
    x = x.drop([
        "date",
        "days_post_first_lockdown",
        "region",
        "numtotal",
        "numdeaths",
        "ratetotal",
        "ratetested"
    ], "columns")

    columns = x.columns
    global scaler
    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(x)
    x = scaler.transform(x)
    return pd.DataFrame(x, columns=columns)


def make_parallel_coordinates_plot(x, cluster, centers, region):
    # For scaling
    # mins = x.min().to_numpy()
    # x = x - mins
    # centers = centers - mins
    # maxs = x.max().to_numpy()
    # x /= maxs
    # centers /= maxs

    x = x.copy()
    x["cluster"] = cluster
    x = x.sort_values("cluster")
    gb = x.groupby("cluster")

    cluster_colors = ["#ff616f", "#26a69a", "#9d46ff"]  # fixed-length
    center_colors = ["#c4001d", "#00766c", "#0a00b6"]  # fixed-length

    for group in gb.groups.keys():
        ax = parallel_coordinates(gb.get_group(group), "cluster", color=cluster_colors[group], linewidth=0.25)
        ax.grid(False)
        ax.set_xticklabels(ax.get_xticklabels(), horizontalalignment="left", rotation="-30")
        plt.plot(ax.get_xticks(), centers[group], color=center_colors[group], label=f"Center of cluster {group}",
                 linewidth=3)
        plt.title(f"Center Vs. Samples for Cluster {group} - Region: {region}")
        plt.xlabel("Features")
        plt.ylabel("Standardized Values")
        leg_txt = plt.legend(loc="upper right").get_texts()[0]
        leg_txt.set_text(f"Samples of cluster {group}")
        plt.show()

    ax = parallel_coordinates(x, "cluster", color=cluster_colors, linewidth=0.25)
    ax.grid(False)
    ax.set_xticklabels(ax.get_xticklabels(), horizontalalignment="left", rotation="-30")

    for i in range(len(centers)):
        plt.plot(ax.get_xticks(), centers[i], color=center_colors[i], label=f"Center of cluster {i}", linewidth=3)

    plt.title(f"Center Vs. Samples for Each Cluster for {region}")
    plt.xlabel("Features")
    plt.ylabel("Standardized Values")
    leg_txts = plt.legend(loc="upper right").get_texts()
    for i in range(len(centers)):
        leg_txts[i].set_text(f"Samples of cluster {i}")

    plt.show()


def make_biplot(x, cluster, centers, region):
    columns = x.columns

    decomposor = PCA(2)
    x = decomposor.fit_transform(x)
    centers = decomposor.transform(centers)
    gb = [x.take(np.where(cluster == value)[0], 0) for value in range(len(centers))]

    cluster_colors = ["#ff616f", "#26a69a", "#9d46ff"]  # fixed-length
    cluster_markers = ["o", "^", "x"]  # fixed-length

    label = "Center of a cluster"
    for i, group in enumerate(gb):
        plt.scatter(group[:, 0], group[:, 1], color=cluster_colors[i], label=f"Samples of cluster {i}",
                    marker=cluster_markers[i])
        plt.plot(centers[:, 0], centers[:, 1], color="None", label=label, linestyle=None, marker="o", mec="black",
                 mfc="gray", ms="20")
        label = None

    mod_components = decomposor.components_ * 8
    label = "Eigenvector"
    for i, column in enumerate(columns):
        x_value = mod_components[0, i]
        y_value = mod_components[1, i]
        plt.arrow(0, 0, x_value, y_value, color="black", head_width=0.2, linewidth=0.1, label=label)
        plt.text(x_value * 1.15, y_value * 1.15, column, ha="center", va="center")
        label = None

    plt.title(f"PCA Biplot With Scaled Eigenvectors for {region}")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.legend(loc="upper right")
    plt.show()


def ingest_data(filename):
    return pd.read_csv(filename, low_memory=False)


def process_data(data):
    train_x = process_features(data)
    test_x = train_x.copy()
    return train_x, test_x


def train_model(train_x):
    model = KMeans(3)
    model.fit(train_x)
    return model


def test_model(test_x, model, region):
    pred_cluster = model.predict(test_x)
    centers = model.cluster_centers_
    make_parallel_coordinates_plot(test_x, pred_cluster, centers, region)
    make_biplot(test_x, pred_cluster, centers, region)


def deploy_model(data, model):
    # x = process_features(pd.read_csv(filename, low_memory=False))
    x = process_features(data)
    return model.predict(x)


if __name__ == "__main__":
    # filename = "../DataPreprocessing/data_by_regions/Canada-2020-12-01.csv"
    # data = ingest_data(filename)
    data, region = dataselector.chooseDataset()
    train_x, test_x = process_data(data)
    model = train_model(train_x)
    test_model(test_x, model, region)
    pred_cluster = deploy_model(data, model)
    print(pred_cluster)

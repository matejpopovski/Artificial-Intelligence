import csv
import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

np.set_printoptions(suppress='True')

def load_data(filepath):
  file = open(filepath, 'r')
  dict_reader = csv.DictReader(file)
  countries_list = list()
  for dict in dict_reader:
      countries_list.append(dict)
  #for dict in countries_list:
      #del dict['']
  return countries_list

def calc_features(row):
  x1 = np.float64(row['Population'])
  x2 = np.float64(row['Net migration'])
  x3 = np.float64(row['GDP ($ per capita)'])
  x4 = np.float64(row['Literacy (%)'])
  x5 = np.float64(row['Phones (per 1000)'])
  x6 = np.float64(row['Infant mortality (per 1000 births)'])
  list_features = np.array([x1, x2, x3, x4, x5, x6])
  return list_features

def distance_between_two_points(first_cluster, second_cluster):
  distance = -1
  for first_point in first_cluster:
      for second_point in second_cluster:
          euclidean_distance = np.linalg.norm(first_point - second_point)
          if euclidean_distance > distance: distance = euclidean_distance
  return distance

def cluster_lookup(clusters, index):
  for cluster in clusters:
      if cluster[0] == index:
          return cluster[1]
  return null

def cluster_remove(clusters, index):
  for cluster in clusters:
      if cluster[0] == index:
          clusters.remove(cluster)
          break

def hac(features):
  feature_length = len(features)
  Z = []
  cluster_list = []
  for i in range(feature_length):
      pair = [i, [features[i]]]
      cluster_list.append(pair)
  curr_idx = feature_length

  while len(cluster_list) >= 2:
      idx1 = -1
      idx2 = -1
      CLD = np.inf
      cluster_size = -1

      for i in range(len(cluster_list)):
          for j in range(i + 1, len(cluster_list)):
              dist = distance_between_two_points(cluster_list[i][1], cluster_list[j][1])
              if dist < CLD:
                  idx1 = cluster_list[i][0]
                  idx2 = cluster_list[j][0]
                  CLD = dist
              elif dist == CLD:
                  if cluster_list[i][0] < idx1:
                      idx1 = cluster_list[i][0]
                      idx2 = cluster_list[j][0]
                      CLD = dist
                      continue
                  elif cluster_list[i][0] == idx1:
                      if cluster_list[j][0] < idx2:
                          idx1 = cluster_list[i][0]
                          idx2 = cluster_list[j][0]
                          CLD = dist
                          continue

      cluster1 = cluster_lookup(cluster_list, idx1)
      cluster2 = cluster_lookup(cluster_list, idx2)
      cluster_merging = []
      for point in cluster1:
          cluster_merging.append(point)
      for point in cluster2:
          cluster_merging.append(point)

      new_pair = [curr_idx, cluster_merging]
      cluster_list.append(new_pair)
      curr_idx += 1

      cluster_size = len(cluster_merging)
      description = [idx1, idx2, CLD, cluster_size]
      Z.append(description)

      cluster_remove(cluster_list, idx1)
      cluster_remove(cluster_list, idx2)
      
  return np.array(Z, dtype=np.float64)

def fig_hac(Z, names):
    fig = plt.figure()
    dendrogram(
        Z,
        leaf_rotation=90.,
        leaf_font_size=8.,
        labels=names
    )
    plt.tight_layout()
    return fig

def normalize_features(features):
    features_array = np.array(features)
    means = np.mean(features_array, axis=0)
    std_devs = np.std(features_array, axis=0)
    normalized_features = (features_array - means) / std_devs
    return [np.array(vec) for vec in normalized_features]

if __name__ == "__main__":

    data = load_data("/content/drive/MyDrive/MATEJ/AI/cs540-projects/hw4-clustering/countries.csv")

    country_names = [row["Country"] for row in data]
    features = [calc_features(row) for row in data]
    features_normalized = normalize_features(features)
    
    n = 50
    Z_raw = hac(features[:n])
    Z_normalized = hac(features_normalized[:n])
    print(Z_normalized)
    fig = fig_hac(Z_raw, country_names[:n])
    # fig = fig_hac(Z_normalized, country_names[:n])
    plt.show()
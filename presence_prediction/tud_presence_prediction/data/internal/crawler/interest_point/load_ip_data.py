import pandas as pd
import numpy as np

from sklearn.cluster import KMeans

from ...utils.data_utils import compute_dist


def get_ip_frame(df, home_coordinate,n_clusters=10 , spherical_dist=True):
    x = np.stack([df['lat'].to_numpy(), df['long'].to_numpy()], axis=1)

    #TODO this assumes a euclidean manifold (no curviture)
    if n_clusters > 0:
        kmeans = KMeans(n_clusters=n_clusters).fit(x)
        distance_2_interst_points = compute_dist(x[None], kmeans.cluster_centers_[:, None, :], spherical_dist=spherical_dist)
        interest_points = kmeans.cluster_centers_

    home = np.array(home_coordinate,dtype=x.dtype)
    distance_2_home = compute_dist(x, home[None, :], spherical_dist=spherical_dist)

    data = dict()
    data['distance_2_home'] = distance_2_home
    for n in range(n_clusters):
        data['distance_2_ip_' + str(n)] = distance_2_interst_points[n]

    ip_df = pd.DataFrame(data=data)
    ip_df['home_coordinates_lat'] = home[0]
    ip_df['home_coordinates_long'] = home[1]
    for n in range(n_clusters):
        ip_df['ip_' + str(n) + '_coordinates_lat'] = interest_points[n, 0]
        ip_df['ip_' + str(n) + '_coordinates_long'] = interest_points[n, 1]

    return ip_df
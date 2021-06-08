

import pandas as pd
import numpy as np

import plotly.offline as plt
import plotly.graph_objs as go

"""
Đối với file 1
    k = 7
Đối với file 2
    k = 3

"""

DATASET_DIR="k_means_clustering_test_1.csv" 
# DATASET_DIR="k_means_clustering_test_2.csv" 

NUM_CLUSTER = 7
# NUM_CLUSTER = 3

temp = None

def k_means_clustering(path, k):
    data = pd.read_csv(path)#Đưa [V1,V2] -> về dataframe
    data = data[['V1', 'V2']]
    k_means = (data.sample(k, replace=False))#Với 7 cụm -> khởi tạo 7 centroids
    k_means2 = pd.DataFrame()
    clusters = pd.DataFrame()
    print('Initial means:\n', k_means)
    
    while not k_means2.equals(k_means):#Lặp cho đến khi số cụm không đổi -> dừng lại.
    
        # distance matrix
        cluster_count = 0
        for idx, k_mean in k_means.iterrows():
            clusters[cluster_count] = (data[k_means.columns] - np.array(k_mean)).pow(2).sum(1).pow(0.5)
            #Mỗi thuộc tính trừ cho mean -> lấy square root
            #Giả sử có 3 thuộc tính thì cộng hết lại
            cluster_count += 1
    
        # print(cluster_count)#Số cụm là 7 -> ta duyệt vài lần thì tiến đến hội tụ.
        # update cluster
        data['MDCluster'] = clusters.idxmin(axis=1)#Ở đây có 2 thuộc tính V1, V2
        # store previous cluster
        k_means2 = k_means
        k_means = pd.DataFrame()
        # print(data.groupby('MDCluster').agg(np.mean))#Tính lại tọa độ của tâm cụm theo 2 thuộc tính V1, V2
        k_means_frame = data.groupby('MDCluster').agg(np.mean)
        
        k_means[k_means_frame.columns] = k_means_frame[k_means_frame.columns]
    
        print(k_means.equals(k_means2))
    global temp
    temp = data
    #Sau khi thu được data frame -> plot ra.
    """plotting
    Bước 1: Vẽ data lên
    Bước 2: Vẽ tâm cụm
    """
    print('Plotting...')
    data_graph = [go.Scatter(
        x=data['V1'],
        y=data['V2'].where(data['MDCluster'] == c),
        mode='markers',
        name='Cluster: ' + str(c)
    ) for c in range(k)]

    data_graph.append(
        go.Scatter(
            x=k_means['V1'],
            y=k_means['V2'],
            mode='markers',
            marker=dict(
                size=10,
                color='#000000',
            ),
            name='Centroids of Clusters'
        )
    )

    plt.plot(data_graph, filename='cluster.html')
        
if __name__=='__main__':
    for i in range (0,5):
        k_means_clustering(DATASET_DIR,NUM_CLUSTER)


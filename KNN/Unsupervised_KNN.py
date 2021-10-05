from sklearn import neighbors
import numpy as np

# ex. 1
# for unsupervised data, use neighbors.NearestNeighbors
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
nbrs = neighbors.NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)
distance, indices = nbrs.kneighbors(X)
print('----------- ex. 1 -----------\n', distance, '\n', indices)
print(distance, '\n', indices)
print(nbrs.kneighbors_graph(X).toarray())

# ex. 2
kdt = neighbors.KDTree(X, leaf_size=30, metric='euclidean')
distance = kdt.query(X, k=2, return_distance=False)
print('----------- ex. 2 -----------\n', distance)



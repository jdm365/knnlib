import pandas as pd
import numpy as np
import faiss
import os
from knnlib import FlatIndexL2, IVFIndex 
from time import perf_counter




def flat_search_knnlib(index_data, query_data, k):
    index = FlatIndexL2(index_data.shape[1])
    index.add(index_data)
    index.train(index_data)

    init = perf_counter()
    results = index.search(query_data, k)
    print('knnlib flat search time: ', perf_counter() - init)

    distances = []
    indices   = []
    for result in results:
        _distances = []
        _indices   = []
        for distance, _idx in result:
            _distances.append(distance)
            _indices.append(_idx)
        distances.append(_distances)
        indices.append(_indices)
    return distances, indices


def flat_search_faiss(index_data, query_data, k):
    index = faiss.IndexFlatL2(index_data.shape[1])
    index.add(index_data)
    index.train(index_data)
    
    init = perf_counter()
    distances, indices = index.search(query_data, k)
    print('knnlib flat search time: ', perf_counter() - init)

    return distances, indices


def ivf_search_knnlib(index_data, query_data, k, n_centroids, n_probe):
    index = IVFIndex(index_data.shape[1], n_centroids, n_probe)
    index.add(index_data)
    index.train(index_data)

    init = perf_counter()
    distances, indices = index.search(query_data, k)
    print('knnlib ivf search time: ', perf_counter() - init)
    print([x for idxs in indices for x in idxs])

    return distances, indices


def ivf_search_faiss(index_data, query_data, k, n_centroids, n_probe):
    index = faiss.IndexIVFFlat(faiss.IndexFlatL2(index_data.shape[1]), index_data.shape[1], n_centroids, faiss.METRIC_L2)
    index.train(index_data)
    index.add(index_data)
    index.nprobe = n_probe

    init = perf_counter()
    distances, indices = index.search(query_data, k)
    print('faiss ivf search time: ', perf_counter() - init)

    return distances, indices


def evaluate_performance(groundtruth, indices):
    # groundtruth: (n_query, k)
    # indices: (n_query, k)
    ## Top groundtruth.shape[1] recall
    n_query = groundtruth.shape[0]
    n_correct = 0
    for idx in range(n_query):
        n_correct += len(set(groundtruth[idx]) & set(indices[idx]))
    return n_correct / (n_query * groundtruth.shape[1])


if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.realpath(__file__))
    data_path = os.path.join(current_dir, '../data/sift/')

    QUERY_PATH = os.path.join(data_path, 'query.npy')
    LEARN_PATH = os.path.join(data_path, 'learn.npy')
    BASE_PATH  = os.path.join(data_path, 'base.npy')
    GROUNDTRUTH_PATH = os.path.join(data_path, 'groundtruth.npy')

    query_data  = np.load(QUERY_PATH)
    learn_data  = np.load(LEARN_PATH)
    base_data   = np.load(BASE_PATH)
    groundtruth = np.load(GROUNDTRUTH_PATH)

    index_data = base_data

    k = 100
    n_centroids = 128 
    n_probe = 8

    # flat search
    #distances, indices = flat_search_knnlib(index_data, query_data, k)
    #distances, indices = flat_search_faiss(index_data, query_data, k)
    distances, indices = ivf_search_knnlib(index_data, query_data, k, n_centroids, n_probe)
    #distances, indices = ivf_search_faiss(index_data, query_data, k, n_centroids, n_probe)

    df = pd.DataFrame({
        'distances': distances,
        'indices': indices
        }).explode(['distances', 'indices'])
    #print(df)

    print(f'flat search top {k} recall: {evaluate_performance(groundtruth, indices)}')

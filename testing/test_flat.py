import numpy as np
import os
import faiss
from time import perf_counter





if __name__ == '__main__':
    dim = 512
    index_size = 1_000_000 
    query_size = 1000


    index_data = np.random.random((index_size, dim)).astype('float32')
    query_data = index_data[:query_size]

    '''
    index = faiss.IndexFlatL2(dim)
    index.add(index_data)

    init_time = perf_counter()
    index.search(query_data, 5)
    end_time = perf_counter()
    print(f'Query time: {end_time - init_time}')
    '''


    # Using a quantizer and IVFFlat for better performance
    nlist = 1
    quantizer = faiss.IndexFlatL2(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)

    index.train(index_data)
    index.add(index_data)
    index.nprobe = 1

    init_time = perf_counter()
    index.search(query_data, 5)
    end_time = perf_counter()
    print(f'Query time: {end_time - init_time}')

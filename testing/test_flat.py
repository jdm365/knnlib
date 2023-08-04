import numpy as np
import faiss
from knnlib import FlatIndexL2, IVFIndex 
from time import perf_counter



if __name__ == '__main__':
    dim = 128
    index_size = 2_000_00
    query_size = 1000

    index_data = np.random.random((index_size, dim)).astype('float32')
    query_data = index_data[:query_size]

    '''
    index = faiss.IndexFlatL2(dim)
    index.add(index_data)

    init_time = perf_counter()
    index.search(query_data, 5)
    end_time = perf_counter()
    print(f'faiss query time: {end_time - init_time}')


    index = FlatIndexL2(dim)
    index.add(index_data)

    init_time = perf_counter()
    result1 = index.search(query_data, 5)
    end_time = perf_counter()
    print(f'knnlib query time: {end_time - init_time}')


    #print(f"FAISS:  {result0}")
    print(f"KNNLIB: {result1}")
    '''

    index = IVFIndex(dim, 4096, 16)
    index.add(index_data)
    index.train(index_data)

    init_time = perf_counter()
    result1 = index.search(query_data, 5)
    end_time = perf_counter()
    print(f'knnlib query time: {end_time - init_time}')

    index = faiss.IndexIVFFlat(faiss.IndexFlatL2(dim), dim, 4096, faiss.METRIC_L2)
    index.train(index_data)
    index.add(index_data)
    index.nprobe = 4096

    init_time = perf_counter()
    result0 = index.search(query_data, 5)
    end_time = perf_counter()
    print(f'faiss query time: {end_time - init_time}')

    #print(f"FAISS:  {result0}")
    #print(f"KNNLIB: {result1}")

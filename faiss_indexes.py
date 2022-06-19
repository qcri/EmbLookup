#GiG

import faiss 
import utils 
import dataset_helpers
import pandas as pd 

class ExactFAISSIndex:
    #dimension corresponds to the number of dimensions of the embedding to be indexed
    def __init__(self, dimension=64):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.mentions = None 
        
    def load_index(self, faiss_index_file_name):
        self.index = faiss.read_index(faiss_index_file_name)

    def add_embedding(self, embeddings):
        self.index.add(embeddings)

    def set_index_to_mention_mapping(self, mentions):
        self.mentions = mentions

    def lookup(self, query_embeddings, k):
        distances, indices = self.index.search(query_embeddings, k)
        words = None 
        if self.mentions is not None:
            words = [self.mentions[i] for i in indices]
        return distances, indices, words

    def save_index(self, output_filename):
        faiss.write_index(self.index, output_filename)

class ApproximateFAISSIndex:
    #dimension corresponds to the number of dimensions of the embedding to be indexed
    # number_of_partitions controls how many voronoi partitions are created for the index
    # Since the index is partitioned, it is possible that some entries are "misplaced"
    # nprobe controls the number of nearby voronoi cells that are also checked 
    def __init__(self, dimension=64, number_of_partitions=128, nprobe=8):
        self.dimension = dimension
        self.number_of_partitions = number_of_partitions
        self.nprobe = nprobe
        self.quantizer = faiss.IndexFlatL2(dimension)
        self.index = faiss.IndexIVFFlat(self.quantizer, dimension, number_of_partitions)
        self.index.nprobe = nprobe

    def load_index(self, faiss_index_file_name):
        self.index = faiss.read_index(faiss_index_file_name)
        #Set the nprobe parameter. It needs some gymnastics as the index is not directly accessible
        ivf = faiss.extract_index_ivf(self.index)
        ivf.nprobe = self.nprobe

    def add_embedding(self, embeddings):
        #IndexIVFFlat needs some embeddings for training and adding
        self.index.train(embeddings)
        self.index.add(embeddings)

    def set_index_to_mention_mapping(self, mentions):
        self.mentions = mentions

    def lookup(self, query_embeddings, k):
        distances, indices = self.index.search(query_embeddings, k)
        words = None 
        if self.mentions is not None:
            words = [self.mentions[i] for i in indices]
        return distances, indices, words

    def save_index(self, output_filename):
        faiss.write_index(self.index, output_filename)

class ApproximateProductQuantizedFAISSIndex:
    #dimension corresponds to the number of dimensions of the embedding to be indexed
    # number_of_partitions controls how many voronoi partitions are created for the index
    # Since the index is partitioned, it is possible that some entries are "misplaced"
    # nprobe controls the number of nearby voronoi cells that are also checked 
    #n_centroids controls the number of centroid ids that will be used to represent the embedding
    # n_bits_per_centroid controls how many bits each centroid will take
    # e.g. n_centroids=8, n_bits_per_centroid=8 results in 8 byte storage for each embedding
    def __init__(self, dimension=64, number_of_partitions=128, nprobe=8, n_centroids=8, n_bits_per_centroid=8):
        self.dimension = dimension
        self.number_of_partitions = number_of_partitions
        self.nprobe = nprobe
        self.n_centroids = n_centroids
        self.n_bits_per_centroid = n_bits_per_centroid

        self.quantizer = faiss.IndexFlatL2(dimension)
        self.index = faiss.IndexIVFPQ(self.quantizer, dimension, number_of_partitions, n_centroids, n_bits_per_centroid)
        self.index.nprobe = nprobe

    def load_index(self, faiss_index_file_name):
        self.index = faiss.read_index(faiss_index_file_name)
        #Set the nprobe parameter. It needs some gymnastics as the index is not directly accessible
        ivf = faiss.extract_index_ivf(self.index)
        ivf.nprobe = self.nprobe

    def add_embedding(self, embeddings):
        #IndexIVFFlat needs some embeddings for training and adding
        self.index.train(embeddings)
        self.index.add(embeddings)

    def set_index_to_mention_mapping(self, mentions):
        self.mentions = mentions

    def lookup(self, query_embeddings, k):
        distances, indices = self.index.search(query_embeddings, k)
        words = None 
        if self.mentions is not None:
            words = [[self.mentions[inner_index] for inner_index in outer_index] for outer_index in indices]
        return distances, indices, words

    def save_index(self, output_filename):
        faiss.write_index(self.index, output_filename)



    
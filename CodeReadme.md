# EmbLookup Quick Start

1. EmbLookup uses fastText for computing semantic similarity. Please download the appropriate word vectors (such as [Common Crawl English](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz)). Please put the unzipped file (such as cc.en.300.bin in the _embeddings_ folder)
2. Please generate the entity mention dataset. It is a CSV file with three columns Index, KGID, and Alias. We provide a preprocessing file (preprocessing.py) that provides a sample implementation for DBPedia. In order to run that, please download [DBPedia Alias list](https://downloads.dbpedia.org/repo/dbpedia/wikidata/alias/2021.02.01/alias.ttl.bz2) and unzip it. Then run preprocess.py. It will generate two files - aliases_processed.csv and kg_index_name_mapping.csv in the appropriate format. It will also update configs.json with appropriate statistics about the dataset.  In the paper, we focused on the entities specified in English. However, our method is language agnostic.
3. Update the configs.json file appropriate hyper parameters such as batch size and number of epochs. We have filled it with the default values used in our experiments.
4. EmbLookup consists of three major operations -- training an embedding model, indexing the entities using the trained model and performing lookup. The file main.py has three functions that provide a sample implementation for each of these operations for the DBPedia dataset. Please modify the parameters of these functions if you want to run it on other datasets.

# Information about Code and Data Files

Please find some basic information about the various files in the folder.

## Code Files
- preprocess.py : It contains two functions. Function process_aliases is customized for DBPedia alias files which can be found [here](https://downloads.dbpedia.org/repo/dbpedia/wikidata/alias/2021.02.01/alias.ttl.bz2). It outputs aliases_processed.csv which is described below. The function map_index_to_entity_name processes aliases_processed.csv to produce kg_index_name_mapping.csv which is also described below. It also updates configs.json with the statistics about the alphabet (such as number of distinct characters etc).
- dataset_helpers.py : It contains two key classes. The class StringDatasetHelper processes files so as to collect statistics about the vocabulary and alphabet. It also has some helper functions such as string_to_tensor that can convert a query into a matrix format that is used the EmbLookup CNN model. The second class KGAliasDataset provides an iterator over the CSV files (aliases_processed.csv or kg_index_name_mapping.csv). For each row, it outputs the entity index and the tensor representation of the string (which can be used to compute syntactic similarity) and fastText embeddings (which measures semantic similarity).
- embedding_learner.py: This file consists of EmbLookupNNModel class and the function train_embedding_model that is used to train it. The class EmbLookupNNModel consists of two PyTorch NN modules. The first _conv_ is a CNN network that takes the matrix outputted by string_to_tensor. It is trained in a manner that the embeddings produces by conv for two strings is proportional to the edit distance between the original strings. The second _combiner_model_ takes the CNN embedding from conv and the fastText embeddings and combines so that the embeddings can handle both syntactic and semantic similarities.
- faiss_indexes.py: This file contains three classes that implement three variants of FAISS that can support exact and approximate similarity search. Each class has the same API that allows it to initialize, add, search along with load/save FAISS indexes.
- utils.py : Contains a potpourri of helper functions.
- main.py : It is a good starting point for trying EmbLookup on DBPedia or other datasets. It has three key functions: setup_and_train_model, index_kg_aliases, LookupFromFAISSIndex. The first takes the output of preprocess.py as input and trains the EmbLookup embedding model. The second function indexes all the entities in  kg_index_name_mapping.csv . The LookupFromFAISSIndex is a convenience class used to query the FAISS index for arbitrary queries. 

## Data Files
- configs.json : This file stores all the necessary configurations. The entry _embedding_model_configs_ specifies the hyper parameters for the embedding model. By default, EmbLookup produces an embedding with 64 dimensions and trained for 32 epochs with a batch size of 128.
- aliases_processed.csv is a CSV file with three columns Index, KGID, and Alias. Index is an integer field that increases consecutively. All rows with the same index correspond to the same entity. The field KGID gives the id of the entity that is unique for the knowledge graph. The Alias is a string representation for the entity. If an entity has 3 aliases, then it will correspond to three rows where the Index and KGID will be the same.
- kg_index_name_mapping.csv is a CSV file with the same structure as aliases_processed.csv. The difference is that it only contains a single entry for each entity. So if an entity has three rows in aliases_processed.csv, we consider the Alias value of first line as the _canonical_ representation for that entity with the Alias values from the second and third line as the Alias. After the EmbLookup embedding model is trained, it only indexes the aliases from kg_index_name_mapping.csv. However, the embeddings are learned so that it can handle all the aliases.
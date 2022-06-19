#GiG

import pandas as pd 
import torch 

import utils 
import dataset_helpers       
import embedding_learner 
import faiss_indexes

def get_kg_alias_data_loader(kg_alias_dataset_file_name, configs, dataset_name):
    batch_size = configs["embedding_model_configs"]["batch_size"]
    string_helper = dataset_helpers.StringDatasetHelper(configs, dataset_name)
    dataset = dataset_helpers.KGAliasDataset(kg_alias_dataset_file_name, string_helper)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
        shuffle=False, num_workers=0, drop_last=False)
    return data_loader

def setup_and_train_model():
    kg_alias_dataset_file_name = "aliases_processed.csv"
    output_file_name="emblookup.pth"
    dataset_name="dataset"

    utils.seed_random_generators(1234)
    configs = utils.load_configs()
    data_loader = get_kg_alias_data_loader(kg_alias_dataset_file_name, configs, dataset_name)

    emblookup_model = embedding_learner.train_embedding_model(configs, data_loader, dataset_name)
    utils.save_emblookup_model(emblookup_model, output_file_name)
    return emblookup_model



def index_kg_aliases():
    kg_alias_mapping_file_name = "kg_index_name_mapping.csv"
    dataset_name="dataset" 
    model_file_name="emblookup.pth"
    faiss_index_file_name="emblookup.findex"

    configs = utils.load_configs()
    data_loader = get_kg_alias_data_loader(kg_alias_mapping_file_name, configs, dataset_name)
    emblookup_model = utils.load_trained_emblookup_model(dataset_name, model_file_name)

    #The following is a memory inefficient approach as we convert all strings to embeddings in one shot
    # this is okay as both approximate and product quantized  faiss indexes need data to "train"
    # before indexing can be done
    # instead of sending a sample to train, we pass all the strings to train and index
    embeddings_list = []
    for step, (entity_index, alias_str_tensor, fasttext_embedding) in enumerate(data_loader):
        emblookup_embeddings = emblookup_model.get_embedding(alias_str_tensor, fasttext_embedding)
        embeddings_list.append(emblookup_embeddings)
    
    emblookup_embeddings = torch.vstack(embeddings_list)

    #Use the default arguments: 64 dimensions
    index = faiss_indexes.ApproximateProductQuantizedFAISSIndex()
    #Convert tensor to numpy as FAISS expects numpy
    index.add_embedding(emblookup_embeddings.numpy())
    index.save_index(faiss_index_file_name)

class LookupFromFAISSIndex:
    def __init__(self):
        self.dataset_name = "dataset"
        self.model_file_name="emblookup.pth"
        self.faiss_index_file_name="emblookup.findex"
        self.mapping_file_name="kg_index_name_mapping.csv"

        #Create an index class with default params
        self.index = faiss_indexes.ApproximateProductQuantizedFAISSIndex()
        self.index.load_index(self.faiss_index_file_name)
        
        self.emblookup_model = utils.load_trained_emblookup_model(self.dataset_name, self.model_file_name)

        configs = utils.load_configs()
        self.string_helper = dataset_helpers.StringDatasetHelper(configs, self.dataset_name)

        #Sometimes the alias/mention can be strings like null which Pandas will convert to np.nan
        # avoid this and read string as is
        df = pd.read_csv(self.mapping_file_name, keep_default_na=False, na_values=[''])
        self.mentions = df["Alias"].tolist()
        df = None 
        self.index.set_index_to_mention_mapping(self.mentions)

        #load fasttext model with default parameters
        self.fasttext_model = utils.load_fasttext_model()

    def lookup(self, query):
        query = query.lower()
        alias_str_tensor = self.string_helper.string_to_tensor(query)
        alias_str_tensor = torch.unsqueeze(alias_str_tensor, dim=0)

        fasttext_embedding = torch.tensor(self.fasttext_model.get_word_vector(query))
        fasttext_embedding = torch.unsqueeze(fasttext_embedding, dim=0)

        embedding = self.emblookup_model.get_embedding(alias_str_tensor, fasttext_embedding)
        distances, indices, words = self.index.lookup(embedding.numpy(), k=5)
        print(f"{query}: {words}")
        print(f"{query}: {indices}")


        



if __name__ == "__main__":
    print("Training EmbLookup model")
    emblookup_model = setup_and_train_model()

    print("Creating FAISS index based on embeddings")
    index_kg_aliases()
    
    print("Looking up with a sample query")
    emblookup = LookupFromFAISSIndex()
    emblookup.lookup("gig")
    
    

   
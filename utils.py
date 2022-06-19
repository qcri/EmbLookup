#GiG

import torch 
import random 
import numpy as np 
import json 

import fasttext
import fasttext.util

import embedding_learner
STATS_SUFFIX = "_alphabet_stats"

def seed_random_generators(seed=1234):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

#Load the english fasttext model and reduce it from 300 dimensions to 64
def load_fasttext_model(file_name="embeddings/cc.en.300.bin", reduced_dimension=64):
    ft = fasttext.load_model(file_name)
    fasttext.util.reduce_model(ft, reduced_dimension)
    return ft

def load_configs(config_file="configs.json"):
    with open(config_file) as json_data_file:
        json_data = json.load(json_data_file)    
    return json_data

#From https://pytorch.org/tutorials/beginner/saving_loading_models.html
def print_model_state_dict_sizes(model):
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

#From https://pytorch.org/tutorials/beginner/saving_loading_models.html
def print_optimizer_state_dict_sizes(optimizer):
    print("Optimizer's state_dict:")
    for var_name in optimizer.state_dict():
        print(var_name, "\t", optimizer.state_dict()[var_name])


def save_emblookup_model(model, output_file_name="emblookup.pth"):
    torch.save(model.state_dict(), output_file_name)

#Assumes that model is an initialized object of type EmbLookupNNModel
#Important: the model does not 
def load_emblookup_model(model, model_file_name="emblookup.pth"):
    model.load_state_dict(torch.load(model_file_name))
    return model

def load_trained_emblookup_model(dataset_name="dataset", model_file_name="emblookup.pth"):
    configs = load_configs()
    emblookup_model = embedding_learner.EmbLookupNNModel(configs, dataset_name)
    emblookup_model = load_emblookup_model(emblookup_model, model_file_name)
    return emblookup_model
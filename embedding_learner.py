#GiG

from sqlalchemy import alias
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from pytorch_metric_learning import miners, losses
import utils

#Code adapted from https://github.com/saravanan-thirumuruganathan/astrid-string-selectivity/blob/master/EmbeddingLearner.py
class EmbLookupNNModel(nn.Module):
    #configs is an object loaded using utils.load_configs
    #dataset is the name of the dataset being trained on so as to get the alphabet related stats
    def __init__(self, configs, dataset_name="dataset"):
        super(EmbLookupNNModel, self).__init__()
        dataset_configs = configs[dataset_name + utils.STATS_SUFFIX]
        embedding_configs = configs["embedding_model_configs"]
        self.embedding_dimension = embedding_configs["embedding_dimension"]
        self.channel_size = embedding_configs["channel_size"]
        self.max_string_length = dataset_configs["max_string_length"]
        self.alphabet_size = dataset_configs["alphabet_size"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.conv = nn.Sequential(
            nn.Conv1d(1, self.channel_size, 3, 1, padding=1, bias=False),
            nn.AvgPool1d(2),
            nn.Conv1d(self.channel_size, self.channel_size, 3, 1, padding=1, bias=False),
            nn.AvgPool1d(2),
            nn.Conv1d(self.channel_size, self.channel_size, 3, 1, padding=1, bias=False),
            nn.AvgPool1d(2),
        )
        # Size after pooling
        # 8 = 2*2*2 = one for each Conv1d layers
        self.flat_size = self.max_string_length // 8 * self.alphabet_size  * self.channel_size
        self.fc1 = nn.Linear(self.flat_size, self.embedding_dimension)

        layer_sizes = [128, 96, 64]
        self.combiner_model = nn.Sequential(
            nn.Linear(layer_sizes[0], layer_sizes[1]),
            nn.ReLU(),
            nn.Dropout(0.001),
            nn.Linear(layer_sizes[1], layer_sizes[2]),
            nn.ReLU(),
        )

        self.conv = self.conv.to(self.device)
        self.fc1 = self.fc1.to(self.device)
        self.combiner_model = self.combiner_model.to(self.device)



    def cnn_forward(self, x):
        N = len(x)
        x = x.view(-1, 1, self.max_string_length)
        x = self.conv(x)
        x = x.view(N, self.flat_size)
        x = self.fc1(x)
        return x

    #Perform convoution to compute the embedding
    def forward(self, alias_str_tensor, fasttext_embedding):
        alias_str_tensor = alias_str_tensor.to(self.device)
        fasttext_embedding = fasttext_embedding.to(self.device)

        edit_distance_embedding = self.cnn_forward(alias_str_tensor)
        overall_embedding = torch.cat((edit_distance_embedding, fasttext_embedding), 1)
        overall_embedding = overall_embedding.to(self.device)
        return self.combiner_model(overall_embedding)

    #This is the inference part of the model where the autograd is turned off
    def get_embedding(self, alias_str_tensor, fasttext_embedding):
        with torch.no_grad():
            #alias_str_tensor is guaranteed to be a tensor but fasttext_embedding is not
            # So convert fasttext_embedding into one
            #fasttext_embedding = torch.from_numpy(fasttext_embedding)
            embedding = self.forward(alias_str_tensor, fasttext_embedding)
            return embedding

def train_embedding_model(configs, data_loader, dataset_name="dataset"):
    max_epochs = configs["embedding_model_configs"]["num_epochs"]
    emblookup_model = EmbLookupNNModel(configs, dataset_name)
    emblookup_model.train()

    miner = miners.MultiSimilarityMiner()
    loss_func = losses.TripletMarginLoss()

    #Use default learning rate
    optimizer = optim.Adam(emblookup_model.parameters())

    for epoch in range(max_epochs):
        running_loss = []
        for step, (entity_index, alias_str_tensor, fasttext_embedding) in enumerate(data_loader):
            optimizer.zero_grad()
            combined_embeddings = emblookup_model(alias_str_tensor, fasttext_embedding)
            hard_pairs = miner(combined_embeddings, entity_index)
            loss = loss_func(combined_embeddings, entity_index, hard_pairs)
            loss.backward()
            optimizer.step()

            running_loss.append(loss.cpu().detach().numpy())
        print(f"Epoch: {epoch+1}/{max_epochs} - Mean Running Loss: {np.mean(running_loss):.4f}")

    return emblookup_model


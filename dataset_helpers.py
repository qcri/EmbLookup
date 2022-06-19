#GiG

import csv 
import torch 
import numpy as np
import utils

#This class is adapted from 
# https://github.com/saravanan-thirumuruganathan/astrid-string-selectivity/blob/master/string_dataset_helpers.py
#This class is used to convert strings into a format processable by the CNN model 
# It can be used in two ways:
# if config_file and dataset_name are not none, then it loads the value from the dataset
# else these values has to be processed from a file using extract_alphabets
class StringDatasetHelper:
    def __init__(self, configs, dataset_name="dataset"):
        self.alphabets = ""
        self.alphabet_size = 0
        self.max_string_length = 0

        alphabet_stats = configs[dataset_name + "_alphabet_stats"] 
        self.alphabets = alphabet_stats["alphabets"]
        self.alphabet_size = alphabet_stats["alphabet_size"]
        self.max_string_length = alphabet_stats["max_string_length"]

    def extract_alphabets(self, file_name):
        f = open(file_name, "r")
        lines = f.read().splitlines()
        f.close()

        self.max_string_length = max(map(len, lines))
        #Make it to even for easier post processing and striding
        if self.max_string_length % 2 != 0:
            self.max_string_length += 1

        #Get the alphabets. Create a set with all strings, sort it and concatenate it
        self.alphabets = "".join(sorted(set().union(*lines)))
        self.alphabet_size = len(self.alphabets)

    #If alphabet is "abc" and for the word abba, this returns [0, 1, 1, 0]
    #0 is the index for a and 1 is the index for b
    def string_to_ids(self, str_val):
        if len(str_val) > self.max_string_length:
            print(f"Warning: long string {str_val} is passed. Subsetting to max length of {self.max_string_length}")
            str_val = str_val[:self.max_string_length]
        indices = [self.alphabets.find(c) for c in str_val]
        if -1 in indices:
            unknown_alphabets = "".join([str_val[index] for index in range(len(indices)) if indices[index] == -1])
            raise ValueError(f"String {str_val} contained unknown alphabets {unknown_alphabets}")
        return indices

    #Given a string (of any length), it outputs a fixed 2D tensor of size alphabet_size * max_string_length
    #If the string is shorter, the rest are filled with zeros
    #Each column corresponds to the i-th character of str_val
    #while each row corresponds to j-th character of self.alphabets
    #This encoding is good for CNN processing
    def string_to_tensor(self, str_val):
        string_indices = self.string_to_ids(str_val)
        one_hot_tensor = np.zeros((self.alphabet_size, self.max_string_length), dtype=np.float32)
        one_hot_tensor[np.array(string_indices), np.arange(len(string_indices))] = 1.0
        return torch.from_numpy(one_hot_tensor)

#This dataset expects a csv file with three columns: Index,KGID,Alias
# Index is an integer, KGID is a unique id for KG
# Alias is a comma separated list of aliases for the id : e.g. "a,b"

#This code assumes that there is no multiple loading (ie only one thread is used by the DataLoader)
class KGAliasDataset(torch.utils.data.IterableDataset):
    #input_file_name: contains the data in the format described above
    #string_helper is an instance of dataset_helpers.StringDatasetHelper
    def __init__(self, input_file_name, string_helper):
        super().__init__()
        self.input_file_name = input_file_name
        self.string_helper = string_helper
        self.fasttext_model = utils.load_fasttext_model()

    
    def __iter__(self):
        f = open(self.input_file_name)
        reader = csv.reader(f)

        #Read and ignore the header line
        next(reader)

        for row in reader:
            entity_index, kg_id, alias = int(row[0]), row[1], row[2]
            
            #We output different variants of a string
            # the original string, one with a character deleted, appended, replaced
            # we also pass the entity_index 
            # the idea is that all strings with the same entity index will form the positive pairs
            alias_str_as_tensor = self.string_helper.string_to_tensor(alias) 


            #fasttext_embedding = np.zeros(64, dtype="float32")
            fasttext_embedding = self.fasttext_model.get_word_vector(alias)
            yield entity_index, alias_str_as_tensor, fasttext_embedding

            #random.choice(string.ascii_letters)
        f.close() 



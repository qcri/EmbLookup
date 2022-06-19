#GiG

import pandas as pd
import csv 
import json 
import re 
import utils

#file_name is the input containing list of aliases such as from https://downloads.dbpedia.org/repo/dbpedia/wikidata/alias/2021.02.01/alias.ttl.bz2
#max_length_to_truncate: remove all aliases with length > max_length_to_truncate
#config_file: json file that stores relevant configs
#dataset_name: stors some stats in the json with key being dataset_name
def process_aliases(input_file_name, output_file_name, max_length_to_truncate=15, config_file="configs.json", dataset_name="dataset"):
    df = pd.read_csv(input_file_name, delimiter=r"\s+", header=None, names=["entity_id_url", "alias_url", "alias_list", "dummy_dot"])

    #Select a subset containing only english items
    df = df[df["alias_list"].str.contains("@en")]

    df["entity_id"] = df["entity_id_url"].str.replace("<http://wikidata.dbpedia.org/resource/", "")
    df["entity_id"] = df["entity_id"].str.replace(">", "")
    #df["new_alias_list"] = df["alias_list"].str.replace("@en", "")
    df["alias_list"] = df["alias_list"].str.replace("@en", "")

    f = open(output_file_name, "w")
    writer = csv.writer(f)
    writer.writerow(["Index","KGID","Alias"])
    write_index = 0 

    #Create a list of distinct alphabets
    alphabets = set()
    #size of the maximum string - this will be <= max_length_to_truncate
    max_valid_string_length = 0
    for row_index, row in df.iterrows():
        increment_index = False
        
        #for elem in row["new_alias_list"].split(","):
        for elem in row["alias_list"].split(","):            
            elem = elem.strip().lower()
            if len(elem) > max_length_to_truncate:
                continue
            #ignore strings containing non ascii characters
            if len(elem) != len(elem.encode()):
                continue
            
            #remove all alphanumeric characters and space 
            orig_elem = elem
            elem = re.sub(r'[^A-Za-z0-9 ]+', '', elem)

            #If the string contained alpha numeric stuff, dont include it.
            if len(elem) != len(orig_elem):
                continue

            #if string contained only spaces ignore them 
            if len(elem.strip()) == 0:
                continue 

            if len(elem) > max_valid_string_length:
                max_valid_string_length = len(elem)

            alphabets = alphabets.union(set(elem))
            writer.writerow([write_index,row['entity_id'],elem])
            increment_index = True
        if increment_index:
            write_index = write_index + 1
        if row_index % 100000 == 0:
            print(f"Processed {row_index} entries.")
    f.close()

     #Make it to even for easier post processing and striding
    if max_valid_string_length % 2 != 0:
        max_valid_string_length = max_valid_string_length + 1 
    alphabet_configs = {"alphabets": "".join(sorted(alphabets)),
                        "max_string_length": max_valid_string_length}
    alphabet_configs["alphabet_size"] = len(alphabet_configs["alphabets"])

    with open(config_file) as json_data_file:
        json_data = json.load(json_data_file)
        json_data[dataset_name + utils.STATS_SUFFIX] = alphabet_configs
    
    print(f"Writing configs to {config_file}:\n {alphabet_configs}")
    with open(config_file, "w") as json_data_file:
        json_formatted_str = json.dumps(json_data, indent=2) + "\n"
        json_data_file.write(json_formatted_str)


#This function takes as input the output of process_aliases
# ie each entry has Index,KGID,Alias fields
#It outputs another file with the same schema except that 
# only the original name of the entity remains
# for e.g. if the same entity has two aliases, the first one is considered as the canonical one
# and dumped into the file
def map_index_to_entity_name(input_file_name, output_file_name):

    input_f = open(input_file_name)
    reader = csv.reader(input_f)
    #ignore the header row 
    next(reader)

    output_f = open(output_file_name, "w")
    writer = csv.writer(output_f)
    writer.writerow(["Index","KGID","Alias"])
    
    prev_entity_index = None
    for row in reader:
        entity_index = row[0]
        #Do not output the alias
        if entity_index == prev_entity_index:
            continue 
        prev_entity_index = entity_index
        writer.writerow(row)

    input_f.close()
    output_f.close()

if __name__ == "__main__":
    #File is downloaded from https://downloads.dbpedia.org/repo/dbpedia/wikidata/alias/2021.02.01/alias.ttl.bz2
    process_aliases("alias.ttl", "aliases_processed.csv")
    map_index_to_entity_name("aliases_processed.csv", "kg_index_name_mapping.csv")
    

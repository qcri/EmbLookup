# EmbLookup

EmbLookup is a Python library that provides an efficient _lookup_ operation over knowledge graph (KG) entities. Specifically, given a keyword _q_, the lookup operation retrieves a set of entities in the KG that are most relevant to _q_. The keyword _q_ may not necessarily match precisely with the text of the relevant entity, for example, due to misspellings. EmbLookup supports fuzzy matching. Additionally, EmbLookup can also retrieve entities that are _semantically_ related to _q_ (such as aliases for the entity that might look syntactically different such as Germany and Deutschland). EmbLookup is based on deep metric learning with triplet loss. We observed that EmbLookup can accelerate entity lookups substantially while being tolerant to many types of errors in the query and data.


# Paper

For details on the architecture of the models used and the training methodology, take a look at our paper Accelerating Entity Lookups in Knowledge Graphs Through Embeddings (ICDE 2022). You can find a copy of the paper [here](https://ashraf.aboulnaga.me/pubs/icde22emblookup.pdf) .

# Support
EmbLookup was developed at HBKU by Saravanan Thirumuruganathan (gmail id is saravanan dot thirumuruganathan ) and Ghadeer Abuoda of HBKU. If you have any queries please contact either one of us.

The [code readme file](CodeReadme.md) should provide a good starting point.

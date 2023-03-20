import os
import argparse
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import csv
from nltk.stem import *
from IPython.display import display

# Useful if you want to perform stemming.
import nltk
stemmer = nltk.stem.PorterStemmer()

categories_file_name = r'/workspace/datasets/product_data/categories/categories_0001_abcat0010000_to_pcmcat99300050000.xml'

queries_file_name = r'/workspace/datasets/train.csv'
output_file_name = r'/workspace/datasets/fasttext/labeled_queries.txt'
output_parent_file_name = r'/workspace/datasets/fasttext/category_parent.txt'


parser = argparse.ArgumentParser(description='Process arguments.')
general = parser.add_argument_group("general")
general.add_argument("--min_queries", default=1,  help="The minimum number of queries per category label (default is 1)")
general.add_argument("--output", default=output_file_name, help="the file to output to")

args = parser.parse_args()
output_file_name = args.output

if args.min_queries:
    min_queries = int(args.min_queries)

# The root category, named Best Buy with id cat00000, doesn't have a parent.
root_category_id = 'cat00000'

tree = ET.parse(categories_file_name)
root = tree.getroot()

# Parse the category XML file to map each category id to its parent category id in a dataframe.
# adding itself here because when rolling up category for root it became null
categories = [root_category_id]
parents = [root_category_id]
for child in root:
    id = child.find('id').text
    cat_path = child.find('path')
    cat_path_ids = [cat.find('id').text for cat in cat_path]
    leaf_id = cat_path_ids[-1]
    #print("leaf_id=" + str(leaf_id) + ", root_cat_id=" + str(root_category_id))
    if leaf_id != root_category_id:
        categories.append(leaf_id)
        #print("categories=" + str(leaf_id))
        parents.append(cat_path_ids[-2])
        #print("parent=" + str(cat_path_ids[-2]))

parents_df = pd.DataFrame(list(zip(categories, parents)), columns =['category', 'parent'])

parents_df.to_csv(output_parent_file_name, index=False)

# Read the training data into pandas, only keeping queries with non-root categories in our category tree.
queries_df = pd.read_csv(queries_file_name)[['category', 'query']]
queries_df = queries_df[queries_df['category'].isin(categories)]

# IMPLEMENT ME: Convert queries to lowercase, and optionally implement other normalization, like stemming.

# IMPLEMENT ME: Roll up categories to ancestors to satisfy the minimum number of queries per category.

stemmer = PorterStemmer()
# Output labeled query data as a space-separated file, making sure that every category is in the taxonomy.
queries_df = queries_df[queries_df['category'].isin(categories)]

# use porter for stemming the query
def stem_query(sentence):
    tokens = sentence.split()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return ' '.join(stemmed_tokens)

queries_df['query'] = queries_df['query'].str.lower().replace(r'[^a-z0-9]', ' ', regex=True).replace(r'\s+', ' ', regex=True).apply(stem_query)

#loop to rollup categories if it doesnt meet the threshold
while True:
    queries_df_with_counts = queries_df.groupby(['category']).size().reset_index(name='count')
    queries_df_with_counts_merged = queries_df.merge(queries_df_with_counts, how='left', on='category').merge(parents_df, how='left', on="category")
    below_threshold_df = queries_df_with_counts_merged[(queries_df_with_counts_merged["count"] < min_queries)]
    #print("below_threshold_df len=" + str(len(below_threshold_df)))
    if len(below_threshold_df) <= 0:
        break

    #print(parents_df.head())

    queries_df_with_counts_merged.loc[queries_df_with_counts_merged['category'].isin(below_threshold_df['category']), 'category'] = queries_df_with_counts_merged['parent']

    queries_df = queries_df_with_counts_merged[['category', 'query']]
    print (queries_df.head())
    print(f"Number of unique categories: {queries_df['category'].nunique()}")

print(f"Number of unique categories: {queries_df['category'].nunique()}")

queries_df.fillna("XXX", inplace=True)
# Create labels in fastText format.
queries_df['label'] = '__label__' + queries_df['category']

# Output labeled query data as a space-separated file, making sure that every category is in the taxonomy.
queries_df = queries_df[queries_df['category'].isin(categories)]
queries_df['output'] = queries_df['label'] + ' ' + queries_df['query']
queries_df[['output']].to_csv(output_file_name, header=False, sep='|', escapechar='\\', quoting=csv.QUOTE_NONE, index=False)

# A simple client for querying driven by user input on the command line.  Has hooks for the various
# weeks (e.g. query understanding).  See the main section at the bottom of the file
from opensearchpy import OpenSearch
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import argparse
import json
import os
from getpass import getpass
from urllib.parse import urljoin
import pandas as pd
import fileinput
import logging
from nltk.stem import *
import nltk
import fasttext

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(format='%(levelname)s:%(message)s')

model = fasttext.load_model("/workspace/search_with_machine_learning_course/week3/model_queries.bin")
stemmer = nltk.stem.PorterStemmer()

# expects clicks and impressions to be in the row
def create_prior_queries_from_group(
        click_group):  # total impressions isn't currently used, but it mayb worthwhile at some point
    click_prior_query = ""
    # Create a string that looks like:  "query": "1065813^100 OR 8371111^89", where the left side is the doc id and the right side is the weight.  In our case, the number of clicks a document received in the training set
    if click_group is not None:
        for item in click_group.itertuples():
            try:
                click_prior_query += "%s^%.3f  " % (item.doc_id, item.clicks / item.num_impressions)

            except KeyError as ke:
                pass  # nothing to do in this case, it just means we can't find priors for this doc
    return click_prior_query


# expects clicks from the raw click logs, so value_counts() are being passed in
def create_prior_queries(doc_ids, doc_id_weights,
                         query_times_seen):  # total impressions isn't currently used, but it mayb worthwhile at some point
    click_prior_query = ""
    # Create a string that looks like:  "query": "1065813^100 OR 8371111^89", where the left side is the doc id and the right side is the weight.  In our case, the number of clicks a document received in the training set
    click_prior_map = ""  # looks like: '1065813':100, '8371111':809
    if doc_ids is not None and doc_id_weights is not None:
        for idx, doc in enumerate(doc_ids):
            try:
                wgt = doc_id_weights[doc]  # This should be the number of clicks or whatever
                click_prior_query += "%s^%.3f  " % (doc, wgt / query_times_seen)
            except KeyError as ke:
                pass  # nothing to do in this case, it just means we can't find priors for this doc
    return click_prior_query


# Hardcoded query here.  Better to use search templates or other query config.
def create_query(user_query, click_prior_query, filters, sort="_score", sortDir="desc", size=10, source=None, isUseSynonyms=False, use_category_filter=False, use_category_boost=False, categories=[]):
    if use_category_filter and categories:
        cat_filter = {
            "terms": {
                "categoryPathIds.keyword": categories
            }
        }
        if not filters:
            filters = cat_filter
        else:
            filters.append(cat_filter)

    query_obj = {
        'size': size,
        "sort": [
            {sort: {"order": sortDir}}
        ],
        "query": {
            "function_score": {
                "query": {
                    "bool": {
                        "must": [

                        ],
                        "should": [  #
                            {
                                "match": {
                                    "name": {
                                        "query": user_query,
                                        "fuzziness": "1",
                                        "prefix_length": 2,
                                        # short words are often acronyms or usually not misspelled, so don't edit
                                        "boost": 0.01
                                    }
                                }
                            },
                            {
                                "match_phrase": {  # near exact phrase match
                                    "name.hyphens": {
                                        "query": user_query,
                                        "slop": 1,
                                        "boost": 50
                                    }
                                }
                            },
                            {
                                "multi_match": {
                                    "query": user_query,
                                    "type": "phrase",
                                    "slop": "6",
                                    "minimum_should_match": "2<75%",
                                    "fields": ["name^10", "name.hyphens^10", "shortDescription^5",
                                               "longDescription^5", "department^0.5", "sku", "manufacturer", "features",
                                               "categoryPath"]
                                }
                            },
                            {
                                "terms": {
                                    # Lots of SKUs in the query logs, boost by it, split on whitespace so we get a list
                                    "sku": user_query.split(),
                                    "boost": 50.0
                                }
                            },
                            {  # lots of products have hyphens in them or other weird casing things like iPad
                                "match": {
                                    "name.hyphens": {
                                        "query": user_query,
                                        "operator": "OR",
                                        "minimum_should_match": "2<75%"
                                    }
                                }
                            }
                        ],
                        "minimum_should_match": 1,
                        "filter": filters  #
                    }
                },
                "boost_mode": "multiply",  # how _score and functions are combined
                "score_mode": "sum",  # how functions are combined
                "functions": [
                    {
                        "filter": {
                            "exists": {
                                "field": "salesRankShortTerm"
                            }
                        },
                        "gauss": {
                            "salesRankShortTerm": {
                                "origin": "1.0",
                                "scale": "100"
                            }
                        }
                    },
                    {
                        "filter": {
                            "exists": {
                                "field": "salesRankMediumTerm"
                            }
                        },
                        "gauss": {
                            "salesRankMediumTerm": {
                                "origin": "1.0",
                                "scale": "1000"
                            }
                        }
                    },
                    {
                        "filter": {
                            "exists": {
                                "field": "salesRankLongTerm"
                            }
                        },
                        "gauss": {
                            "salesRankLongTerm": {
                                "origin": "1.0",
                                "scale": "1000"
                            }
                        }
                    },
                    {
                        "script_score": {
                            "script": "0.0001"
                        }
                    }
                ]

            }
        }
    }
    if click_prior_query is not None and click_prior_query != "":
        query_obj["query"]["function_score"]["query"]["bool"]["should"].append({
            "query_string": {
                # This may feel like cheating, but it's really not, esp. in ecommerce where you have all this prior data,  You just can't let the test clicks leak in, which is why we split on date
                "query": click_prior_query,
                "fields": ["_id"]
            }
        })

    if isUseSynonyms:
        #hardcoded to first one
        query_obj["query"]["function_score"]["query"]["bool"]["should"][0] = {
            "match": {
                                    "name.synonyms": {
                                        "query": user_query,
                                        "fuzziness": "1",
                                        "prefix_length": 2,
                                        # short words are often acronyms or usually not misspelled, so don't edit
                                        "boost": 0.01
                                    }
                                }
        }

    if use_category_boost:
        cat_boost = {
            "terms": {
                "categoryPathIds.keyword": categories,
                "boost": 10.0
            }
        }
        query_obj["query"]["function_score"]["query"]["bool"]["should"].append(cat_boost)
    

    if user_query == "*" or user_query == "#":
        # replace the bool
        try:
            query_obj["query"] = {"match_all": {}}
        except:
            print("Couldn't replace query for *")

    if source is not None:  # otherwise use the default and retrieve all source
        query_obj["_source"] = source

    print ("query_obj=" + str(query_obj))
    return query_obj

def get_predicted_cat(query: str):
    # use porter for stemming the query
    stemmed_user_query = query.lower()
    tokens = stemmed_user_query.split()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    stemmed_user_query = ' '.join(stemmed_tokens)

    k = 5
    cats, probs = model.predict(stemmed_user_query, k)
    print(cats)
    print(probs)

    cat_len = len(cats)
    predicted_cats = []
    category_threshold = 0.3
    print("len="+str(cat_len))

    #cut off by threshold
    for i in range (0, cat_len):
        #print("i="+ str(i) + ", label=" + cats[i] + ", prob=" + str(probs[i]))
        if probs[i] > category_threshold:
            cur = cats[i].replace("__label__", "")
            predicted_cats.append(cur)
            #print("adding cat="+ cur)

    return predicted_cats
    

def search(client, user_query, index="bbuy_products", sort="_score", sortDir="desc", isUseSynonyms=False, use_category_filter=False, use_category_boost=False):
    #### W3: classify the query
    predicted_cats = []
    if use_category_filter or use_category_boost:
        predicted_cats = get_predicted_cat(user_query)
        print("predicted_cats=")
        print(predicted_cats)

    #### W3: create filters and boosts
    query_obj = create_query(user_query, click_prior_query=None, filters=None, sort=sort, sortDir=sortDir, source=["name", "shortDescription"], isUseSynonyms=isUseSynonyms, use_category_filter=use_category_filter, use_category_boost=use_category_boost, categories=predicted_cats)
    logging.info(query_obj)
    response = client.search(query_obj, index=index)
    if response and response['hits']['hits'] and len(response['hits']['hits']) > 0:
        hits = response['hits']['hits']
        print(json.dumps(response, indent=2))


if __name__ == "__main__":
    host = 'localhost'
    port = 9200
    isUseSynonyms = False
    auth = ('admin', 'admin')  # For testing only. Don't store credentials in code.
    parser = argparse.ArgumentParser(description='Build LTR.')
    general = parser.add_argument_group("general")
    #parser.add_argument('--synonyms', action="store_true",
    #                     help='If this is set, opnesearch will use the name.synonyms field')
    general.add_argument("-i", '--index', default="bbuy_products",
                         help='The name of the main index to search')
    general.add_argument("-s", '--host', default="localhost",
                         help='The OpenSearch host name')
    general.add_argument("-p", '--port', type=int, default=9200,
                         help='The OpenSearch port')
    general.add_argument('--user',
                         help='The OpenSearch admin.  If this is set, the program will prompt for password too. If not set, use default of admin/admin')
    general.add_argument('--synonyms', default=False, action="store_true",
                         help='If this is set, opnesearch will use the name.synonyms field')
    general.add_argument('--use_category_filter', default=False, action="store_true",
                         help='If this is set, the results will be filtered by predicted category over threshold')
    general.add_argument('--use_category_boost', default=False, action="store_true",
                         help='If this is set, the predicted category will be boosted')

    args = parser.parse_args()

    if len(vars(args)) == 0:
        parser.print_usage()
        exit()

    host = args.host
    port = args.port
    isUseSynonyms = args.synonyms
    use_category_boost = args.use_category_boost
    use_category_filter = args.use_category_filter
    #print("synonuyms flag =" + str(args.synonyms))
    if args.user:
        password = getpass()
        auth = (args.user, password)

    base_url = "https://{}:{}/".format(host, port)
    opensearch = OpenSearch(
        hosts=[{'host': host, 'port': port}],
        http_compress=True,  # enables gzip compression for request bodies
        http_auth=auth,
        # client_cert = client_cert_path,
        # client_key = client_key_path,
        use_ssl=True,
        verify_certs=False,  # set to true if you have certs
        ssl_assert_hostname=False,
        ssl_show_warn=False,

    )
    index_name = args.index
    query_prompt = "\nEnter your query (type 'Exit' to exit or hit ctrl-c):"
    print(query_prompt)
    while True:
        query = input(query_prompt).rstrip()
        if query == "Exit":
            break

        search(client=opensearch, user_query=query, index=index_name, isUseSynonyms=isUseSynonyms, use_category_filter=use_category_filter, use_category_boost=use_category_boost)

        print(query_prompt)

    
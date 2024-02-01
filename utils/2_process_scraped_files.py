# USAGE
# get_only_bodies(insert the filepath where the "Reddit_scraping.py" file saved the raw json)
# The function outputs a .json file with 10000 bodies of Reddit posts/comments
# Remember to change filename before running for the other community/subreddit .json file



import json
import nltk
from nltk import word_tokenize

def get_only_N_bodies(path):
    
# oper the raw scraped json dictionary with everything inside
    with open(path, encoding = "utf-8-sig", mode = 'r') as handle:
        json_data = [json.loads(line) for line in handle]

    bodies = []
    for item in json_data:
        bodies.append(item.get('author_flair_text'))
    
    keep only the first 10000 documents
    bodies = bodies[:10000]
    
    # save the final file in the current directory
    out_file = open("prolife_bodies.json", "w")
    json.dump(bodies, out_file)
    out_file.close()
    
    return bodies

#insert the filepath where 1_Reddit_scraping.py saved the raw file
get_only_N_bodies("./filepath")

"""
Simple script to create a table of character and narrator speech percentages on the annotated data. 
"""

# === Imports === 

from os.path import join, basename
import glob
import pandas as pd
import numpy as np
import re



# === Parameters === 

variants = ["first", "third"] # first|third
resultsfile = "ELTeC-fra_character-and-narrator-speech-proportions.tsv"



# === Functions === 

def load_annotated(annotated): 
    with open(annotated, "r", encoding="utf8") as infile: 
        data = pd.read_csv(infile, sep="\t")
    return data



def get_props(data, variant): 

    # Get the proportion of sentences in narrator and character speech
    total = len(data["label_text"])
    csents = len([entry for entry in data["label_text"] if entry == "character"])
    nsents = len([entry for entry in data["label_text"] if entry == "narrator"])
    csentsp = csents / total * 100
    nsentsp = nsents / total * 100

    # Get the proportion of words in narrator and character speech 
    words = {}
    data_split = data.groupby(by="label_text")
    for kind,data in data_split: 
        text = " ".join(data["text"].tolist())
        words[kind] = len(re.split("\W+", text))
    total = words["character"] + words["narrator"]
    cwordsp = words["character"] / total * 100
    nwordsp = words["narrator"] / total * 100

    # Collect all the information in one dict
    props = {
        "narration" : variant,
        "char-sents" : np.round(csentsp,2),
        "narr-sents" : np.round(nsentsp,2),
        "char-words" : np.round(cwordsp,2),
        "narr-words" : np.round(nwordsp,2),
        }
    return props



def save_results(results): 
    pd.options.display.float_format = '{:.2f}'.format
    results = pd.DataFrame(results).T 
    print(results)
    with open(resultsfile, "w", encoding="utf8") as outfile: 
        results.to_csv(outfile, sep="\t")



# === Main === 

def main(): 
    results = {}
    for variant in variants: 
        annotated_folder = join("4-annotated", variant, "*.tsv")
        for annotated in glob.glob(annotated_folder): 
            idno = basename(annotated).split(".")[0]
            data = load_annotated(annotated)
            results[idno] = get_props(data, variant)
    save_results(results)

main()
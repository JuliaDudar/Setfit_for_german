import spacy
import re
import pandas as pd
from os.path import join, basename
import glob
import csv

# Get the list of filenames
filenames = glob.glob(join("2-plaintext", 'third', "*.txt"))
resultfolder = "D:/Setfit/setfit-main/plain-splitted"

def apply_to_plain(filename):
    # Read the content of the file
    with open(filename, "r", encoding="utf8") as infile:
        ptext = infile.read()[:50000]  # Limiting the text for testing purposes

    # Load the SpaCy model
    nlp = spacy.load("de_dep_news_trf")
    nlp.max_length = 2000000

    # Process the text using SpaCy
    doc = nlp(ptext)

    # Extract sentences, clean them, and store them in a list
    sents = [sent.text.strip() for sent in doc.sents if len(sent.text) > 5]

    # Create a DataFrame with columns "text", "label", "label_text"
    results = pd.DataFrame({
        "text": sents,        # Sentences go in the "text" column
        "label": "",          # Empty "label" column
        "label_text": ""      # Empty "label_text" column
    })

    # Save the DataFrame to a TSV file without extra quotation marks
    output_filename = join(resultfolder, basename(filename)[:-4] + ".csv")
    results.to_csv(output_filename, sep="\t", index=True, encoding="utf8",)

# Process each file
for filename in filenames:
    print(f"Working on {basename(filename)}...", end=" ", flush=True)
    apply_to_plain(filename)
    print("Done")

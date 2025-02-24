"""
Script to test SetFit for German language data. 

- See: https://huggingface.co/docs/setfit/v1.0.3/en/quickstart
- Prepare: `pip install setfit` 
- The example data used here are sentences in direct or narrator speech (= transfer to literary studies) in German. 
- For this, custom-built local dataset was used. 

Source of training data: ELTeC-: 
https://github.com/COST-ELTeC/ELTeC-deu/tree/master/level1

- Later on, more texts were added, annotated automatically, then corrected and added to the training data.  
"""


# === Imports ===  

# Generic
import pandas as pd 
from collections import Counter 
from os.path import join, basename
import re
import numpy as np
import glob

# Specific
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset, Dataset, DatasetDict
from setfit import SetFitModel, SetFitTrainer, sample_dataset, TrainingArguments, Trainer
import spacy 



# === Parameters === 

# Type of run: train or apply
runtype = "train"
# Variant: First-person data and model or third-person data and model
variant = "third" # third|first

# Training data
trainingdata = glob.glob(join("1-trainingdata", variant, "*.tsv"))

# Model 
model_pretrained = join("JoBeer/german-semantic-base") #"5-models", "german-semantic-base_"+variant+"_2024-09-03_5") #"JoBeer/german-semantic-base"
model_finetuned = join("5-models", "german-semantic-base_"+variant+"_2024-10-23")
num_samples = 50
batch_size = 8
num_epochs = 5

# Evaluation
evaluation_file = join("3-evaluation", variant, "text-pred-true_"+variant+".tsv")

# Application to plain text
plain_filenames = glob.glob(join("2-plaintext", variant, "*.txt"))

# Annotated folder  
annotated_folder = join("4-annotated", variant, "")



# === System operations === 

# Empty GPU / CUDA cache to clear GPU memory
import torch
torch.cuda.empty_cache()



# === Functions === 


def prepare_dataset(): 

    # Load the dataset from the CSV files
    dataframes = []
    for item in trainingdata: 
        print(f"Working on {item}...", flush=True)
        with open(item, "r", encoding="utf8") as infile: 
            item_df = pd.read_csv(infile, sep="\t")
        dataframes.append(item_df)
    dataframes = pd.concat(dataframes, ignore_index=True)

    # Transform into Huggingface DataSetDict with train/test split
    dataset = Dataset.from_pandas(dataframes)
    train_test_split = dataset.train_test_split(test_size=0.5)
    dataset = DatasetDict({
        'train': train_test_split['train'],
        'test': train_test_split['test']
        })

    # Check dataset format
    #print(dataset)

    # Inspect the training data 
    print("\n === Training data ===")
    print(Counter(dataset["train"]["label_text"]))
    #for item in dataset["train"]: 
    #    print(f"{item['text']} ==> {item['label_text']}")


    # Inspect the test data 
    print("\n === Test data ===")
    print(Counter(dataset["test"]["label_text"]))
    #for item in dataset["test"]: 
    #    print(f"{item['text']} ==> {item['label_text']}")

    return dataset
    


def train(dataset): 

    # Load the model
    model = SetFitModel.from_pretrained(model_pretrained)

    # Select n examples per class
    train_dataset = sample_dataset(dataset["train"], label_column="label", num_samples=num_samples)

    # Apply the labels from the dataset to the model
    model.labels = ["narrator", "character"]

    # Set training arguments 
    args = TrainingArguments(
        batch_size=batch_size,      # changed from 32! 
        num_epochs=num_epochs,      # changed from 10!  
        )

    # Instantiate a SetFitTrainer.
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        )

    # Train
    trainer.train()

    # Evaluate on test part of dataset
    test_dataset = dataset["test"]
    metrics = trainer.evaluate(test_dataset)
    print("\n=== Evaluation ===\n", metrics)

    # Save the model 
    model.save_pretrained(model_finetuned) 



def apply_to_test(dataset): 
          
    # Load the previously trained and saved model
    model = SetFitModel.from_pretrained(model_finetuned)

    # Get unlabeled sentences and true labels from test dataset
    unlabeled_sents = list(dataset["test"]["text"])
    #print(unlabeled_sents)
    trues = list(dataset["test"]["label_text"])
    
    # Apply the model to the unlabeled sentences
    preds = model.predict(unlabeled_sents)

    # Check accuracy 
    correct_count = 0
    for item1, item2 in zip(preds, trues):
        if item1 == item2:
            correct_count += 1
    accuracy = np.round(correct_count / len(preds),2)
    print(f"\nAccuracy: {accuracy}.")
   
    # Save text, predictions and true labels to CSV 
    sents = pd.Series(unlabeled_sents, name="text") 
    preds = pd.Series(preds, name="pred")
    trues = pd.Series(trues, name="true") 
    results = pd.DataFrame([unlabeled_sents, preds, trues], index=["text", "pred", "true"]).T
    print(results.head())
    with open(evaluation_file, "w", encoding="utf8") as outfile: 
        results.to_csv(outfile, sep="\t")
    


def apply_to_plain(filename): 

    # Load a plain text file 
    with open(filename, "r", encoding="utf8") as infile: 
        ptext = infile.read()[:50000] # For testing
    
    # Sentence splitting using spacy 
    nlp = spacy.load("de_dep_news_trf")
    nlp.max_length = 2000000
    doc = nlp(ptext)
    sents = [sent.text for sent in doc.sents if len(sent.text) > 5]
    sents = [re.sub("\n", "", sent) for sent in sents]
    # TODO: select all or random selection

    # Load and apply the model to the unlabeled sentences
    model = SetFitModel.from_pretrained(model_finetuned)
    preds = model.predict(sents)
    torch.cuda.empty_cache()
    
    # Save text and predicted labels to CSV 
    sents = pd.Series(sents, name="text") 
    preds = pd.Series(preds, name="label_text")
    labels = pd.Series([0 if label == "narrator" else 1 for label in preds], name="label")
    results = pd.DataFrame([sents, labels, preds]).T
    #print(results.head())
    with open(join(annotated_folder, basename(filename)[:-4]+".tsv"), "w", encoding="utf8") as outfile:
        results.to_csv(outfile, sep="\t")
    
    # Calculate proportion of sentence types 
    counts = Counter(preds) 
    total = sum(counts.values())
    percs = {item: (count / total) * 100 for item, count in counts.items()}
    print(f": character {np.round(percs['character'],1)}% – narrator {np.round(percs['narrator'],1)}%.", end="\n")



# === Main === 

def main(): 

    # For fine-tuning a model using training data
    if runtype == "train":
        dataset = prepare_dataset()
        train(dataset)
        apply_to_test(dataset)

    # For applying a fine-tuned model to new plain text
    elif runtype == "apply": 
        for filename in plain_filenames: 
            print(f"Working on {basename(filename)}...", end=" ", flush=True)
            apply_to_plain(filename)
main()


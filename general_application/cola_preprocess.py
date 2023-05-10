# Prep CoLA dataset by combining both the training and validation datasets
# And append a column that determines if it's validation ("test") or training

import pandas as pd


# Pull in data
train_df = pd.read_csv("Data/in_domain_train.tsv", sep="\t", header=None)
test_df = pd.read_csv("Data/in_domain_dev.tsv", sep="\t", header=None)

# Fix columns and add index
train_df.columns = ["source", "label", "label_human", "text"]
train_df["test_index"] = 0

test_df.columns = ["source", "label", "label_human", "text"]
test_df["test_index"] = 1

# Merge dfs and save
final_df = pd.concat([train_df, test_df]).filter(["label", "text", "test_index"])
final_df.to_csv("data/CoLA_clean.csv", index=False)
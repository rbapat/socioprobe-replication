# Prep CoLA dataset by combining both the training and validation datasets
# And append a column that determines if it's validation ("test") or training
# Lastly, do some general EDA to output shape of file

import pandas as pd
import plotly.express as px


def main():
    """Clean data and perform some quick EDA
    """
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
    final_df.to_csv("data/cola_clean.csv", index=False)

    df = final_df

    # EDA process
    # Labels - always sorted
    print(f'Labels: {sorted(set(df["label"]))}')

    # Shape
    print(f"Data shape: {df.shape[0]} rows with {df.shape[1]} columns")
    num_rows = df.shape[0]

    # Print the head of the file
    print(df.head())

    # Group labels into counts
    label_freq = pd.DataFrame(df["label"].value_counts(sort=False, ascending=True)).reset_index()
    label_freq.columns = ["label", "count"]
    label_freq["perc"] = label_freq["count"] / num_rows
    print(label_freq)

    fig = px.bar(label_freq, x="label", y="count")
    fig.update_layout(title="Distribution of Labels")
    fig.update_xaxes(type="category")
    fig.show()


if __name__ == "__main__":
    main()

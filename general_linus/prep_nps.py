## File to extract and clean NPS chat corpus data
from nltk.corpus import nps_chat as nps
import nltk
import pandas as pd
import plotly.express as px
import re
# Also need module nbformat


def extract_age(text: str) -> int:
    """Given the user string, extract the age

    Parameters
    ----------
    string : str
        string containing the user

    Returns
    -------
    int
        value of (10, 20, 30, 40) representing the age of someone
    """
    # age = re.search("\d{2}", string)
    age = re.findall("\d{2}-\d{1,2}-(.*)User\d*", text)[0]

    # Convert age to be 10, 20, etc.
    if "teens" in age:
        age = 10
    elif "adults" in age:
        # This group needs to be removed, so list as 404
        age = 404
    else:
        age = re.search("\d+", age).group(0)

    return int(age)


def clean_username(text: str) -> str:
    """change all usernames to "USER"

    Parameters
    ----------
    text : str
        text body that contains the username

    Returns
    -------
    str
        text body with modified username
    """
    clean_text = re.sub("\d{2}-\d{1,2}-.*User\d*", "USER", text)

    return clean_text


def main():
    seed_val = 5

    # Ensure that NPS chat data is saved
    nltk.download("nps_chat")

    # Pull important columns from data
    data = []
    for p in nps.xml_posts():
        # Modified from https://stackoverflow.com/questions/48135562/convert-xml-to-pandas-data-framework
        data.append({"class": p.get("class"), "text": clean_username(p.text),
                    "label": extract_age(p.attrib["user"])})
    df = pd.DataFrame.from_dict(data)

    # Remove "System" classes - relate to actions that can be done in the chatroom, such as leaving or joining
    df = df[df["class"] != "System"]

    # Remove adults (those we labeled as "50")
    df = df[df["label"] != 404].filter(["text", "label"])
    df = df.reset_index(drop=True)
    df = df.sample(frac=1, random_state=5)

    # # Save dataset to nps_chat_corpus_clean.csv
    df.to_csv("Data/nps_chat_corpus_clean.csv", index=False)
    # print(df)

    # Lastly, show some stats about the dataframe

    # Dataframe head, dimension, & labels
    df = df.sample(frac=1, random_state=seed_val).reset_index(drop=True)
    print(df.head())
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    print(f'Labels: {sorted(set(df["label"]))}')

    # Dataframe label distributions
    label_freq = pd.DataFrame(df["label"].value_counts(sort=False, ascending=True)).reset_index()
    label_freq.columns = ["label", "count"]
    label_freq["perc"] = label_freq["count"] / df.shape[0]
    print(label_freq)

    fig = px.bar(label_freq, x="label", y="count")
    fig.update_layout(title="Distribution of Labels")
    fig.update_xaxes(type="category")
    fig.show()


if __name__ == "__main__":
    main()

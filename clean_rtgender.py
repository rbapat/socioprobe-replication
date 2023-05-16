# Imports
import argparse
import pandas as pd
import numpy as np
import random
from transformers import set_seed, AutoTokenizer

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    return parser.parse_args()

def view_label_pct(df, col='op_gender'):
    print(df[col].value_counts(normalize=True))

def rm_missing_vals(df, label='op_gender', text='post_text'):
    print('\nRemoving Missing Values')
    print(f'Original Shape: {df.shape}')
    print(f'Missing Label: {df[label].isna().sum()}')
    print(f'Missing Text: {df[text].isna().sum()}')
    df = df.dropna()
    print(f'Cleaned Shape: {df.shape}')
    return df

def rm_empty_strings(df, label='op_gender', text='post_text'):
    print('\nRemoving Empty Strings')
    print(f'Original Shape: {df.shape}')
    print(f'Missing Label: {df[df[label]== ""].shape[0]}')
    print(f'Missing Text: {df[df[text] == ""].shape[0]}')
    df = df[(df[label]!= "") & (df[text]!= "")]
    print(f'Cleaned Shape: {df.shape}')
    return df

def filter_and_rename_cols(df, label='op_gender', text='post_text'):
    print('\nFiltering to "Text" and "Label" columns')
    return df[[label, text]].rename(columns = {label: 'label', text: 'text'})

def rm_duplicate_posts(df, cols):
    print('\nRemoving Duplicate Text')
    print(f'Original Shape: {df.shape}')
    df = df.drop_duplicates(cols, keep='first')
    print(f'Cleaned Shape: {df.shape}')
    return df

def labels_to_int(df, label='label'):
    df[label] = df[label].map(lambda x: 0 if x=='M' else 1)

def get_word_count(x, model = 'bert-base-cased'):
    tokenizer = AutoTokenizer.from_pretrained(model)
    return len(tokenizer.tokenize(x, add_special_tokens=True))
    #return len(x.split())

def rm_long_sequence(df, text_col = 'text', model = 'bert-base-cased'): 
    print('\nChecking for Invalid Sequence Lengths')
    print(f'Original Shape: {df.shape}')
    df['word_count'] = 0
    for i in range(0, df.shape[0]):
        if i%100 == 0:
            print(f'Tokenizing Post: {i+1}/{df.shape[0]+1}')

        seq_len = get_word_count(df.loc[i, text_col])

        if seq_len <= 512:
            df.loc[i, 'word_count'] = get_word_count(df.loc[i, text_col])
        else:
            df.loc[i, 'word_count'] = 513
    df = df[df['word_count'] <= 512]
    print(f'Cleaned Shape: {df.shape}')
    return df


def clean_data(df, label='op_gender', text='post_text'):
    df = rm_missing_vals(df, label, text)
    df = rm_empty_strings(df, label, text)
    df = rm_duplicate_posts(df, text)
    df = filter_and_rename_cols(df, label, text)
    labels_to_int(df)
    
    return df[['text', 'label']]

def shuffle_and_sample(df, n, col='label', male=1, female=0):
    # filter data by label and sample n/2 rows
    df_m = df[df[col] == male].sample(int(n/2)).reset_index(drop=True)
    df_f = df[df[col] == female].sample(int(n/2)).reset_index(drop=True)
    
    # return combined df
    return pd.concat([df_m, df_f]).reset_index(drop=True)
    
def main():
    # Parse Args
    args = get_args()

    # Reproducibility
    seed_val = 685
    random.seed(seed_val)
    np.random.seed(seed_val)
    set_seed(seed_val)

    # Data
    data = pd.read_csv(f'data/{args.dataset}.csv')

    print('\nViewing Raw Data Shape')
    print(data.shape)

    # EDA of Raw Data
    print('\nViewing Label Distribution')
    print(view_label_pct(data))

    # Clean Data
    data = clean_data(data)

    # Sample Subset of Data
    # To reduce compute, first randomly sample 21000 posts
    # Then, remove long sequences and re-sample 20000 posts
    # Note, sampling to have equal classes!
    data = shuffle_and_sample(data, 21000)
    data = rm_long_sequence(data).sample(20000)[['text', 'label']]
    
    print('\nFinal Cleaned Data:')
    print(data.shape)
    print(view_label_pct(data, col='label'))
    print(data.head())

    data.to_csv(f'data/{args.dataset}_clean.csv', index=False)

if __name__ == '__main__':
    main()
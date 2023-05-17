import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import tqdm
import torch
import transformers as trfm
from torch.utils.data import Dataset
import pandas as pd


@dataclass
class ProbingDataset(Dataset):
    pooled_embeddings: torch.Tensor
    ground_truth: torch.Tensor
    indices: torch.Tensor
    orig_strings: List[str]

    def get_hidden_dim(self) -> int:
        """Gets the size of each pooled embedding

        Returns:
            int: size of each pooled embedding
        """
        return self.pooled_embeddings.shape[1]

    def get_output_dim(self) -> int:
        """Gets the size of each pooled embedding

        Returns:
            int: size of each pooled embedding
        """
        return len(torch.unique(self.ground_truth))

    def __len__(self):
        return len(self.pooled_embeddings)

    def __getitem__(self, idx):
        return (self.pooled_embeddings[idx], self.ground_truth[idx], self.indices[idx])


def get_model(
    model_type: str, device: torch.device
) -> Tuple[trfm.PreTrainedTokenizer, trfm.PreTrainedModel, trfm.PretrainedConfig]:
    """Configures and creates the required transformers from huggingface

    Args:
        model_type (str): type of huggingface model we are creating
        device (torch.device): what torch device (cuda or cpu) are we using

    Returns:
        Tuple[trfm.PreTrainedTokenizer, trfm.PreTrainedModel, trfm.PretrainedConfig]: tokenizer, model, and model config
    """
    model_config = trfm.AutoConfig.from_pretrained(
        model_type, output_hidden_states=True, output_attentions=True
    )
    tokenizer = trfm.AutoTokenizer.from_pretrained(model_type)
    model = trfm.AutoModel.from_pretrained(model_type, config=model_config)

    return tokenizer, model.to(device), model_config


def create_embeddings(
    all_reviews: List[str],
    emb_path: str,
    model_type: str,
    emb_layer: int,
    device: torch.device,
    batch_size=8,
) -> torch.Tensor:
    """Converts a dataset of raw reviews into the corresponding embedding output or pooled hidden output

    Args:
        all_reviews (List[str]): list of raw reviews to process
        emb_path (str): path to embedding file to write newly created embeddings to
        model_type (str): type of huggingface model to use
        emb_layer (int): what layer of the transformer we want embeddings from. When this is 0, we are using the final output. Otherwise, we will be using the specified (1-indexed) hidden layer output
        device (torch.device): what torch device (cuda or cpu) are we using
        batch_size (int, optional): Size of batches in which data is chunked and processed. Defaults to 8.

    Returns:
        torch.Tensor: tensor of shape `(num_samples, hidden_size)` where num_samples is the number of reviews in `all_reviews`
    """
    tokenizer, model, config = get_model(model_type, device)

    # emb_layer == 0: output embedding of model
    # emb_layer \in [1, 12]: output of nth hidden layer
    assert emb_layer >= 0 and emb_layer <= config.num_hidden_layers

    num_reviews = len(all_reviews)
    pooled_embs = torch.zeros(num_reviews, config.hidden_size, device=device)

    # TODO: refactor this to separate into its own function
    with torch.no_grad():
        model.eval()

        text = f"hidden layer {emb_layer}" if emb_layer > 0 else "output embedding"
        print(f"Creating model embeddings for {text}")

        with tqdm.tqdm(total=num_reviews) as pbar:
            for start in range(0, num_reviews, batch_size):
                end = min(start + batch_size, num_reviews)
                batch = all_reviews[start:end]

                tokenized_text = tokenizer(
                    batch,
                    return_tensors="pt",
                    return_attention_mask=True,
                    padding=True,
                    truncation=True,
                    max_length=config.max_position_embeddings,
                ).to(device)

                embeddings = model(**tokenized_text)

                attention_mask = tokenized_text.attention_mask
                token_embeddings = embeddings.hidden_states[emb_layer]

                # TODO: double check that this actually works, it should zero out the mask for the [CLS] and [SEP]
                special_tokens = [101, 102]  # [CLS], [SEP]
                for tok_id in special_tokens:
                    attention_mask[tokenized_text["input_ids"] == tok_id] = 0

                # https://stackoverflow.com/a/73639621
                input_mask_expanded = (
                    attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                )
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                sum_mask = input_mask_expanded.sum(1)
                sum_mask = torch.clamp(sum_mask, min=1e-9)
                pooled = sum_embeddings / sum_mask

                pooled_embs[start:end, :] = pooled
                pbar.update(end - start)

    os.makedirs(os.path.dirname(emb_path), exist_ok=True)
    torch.save(pooled_embs, emb_path)

    return pooled_embs


def get_dataset(
    data_path: str,
    model_type: str,
    emb_layer: int,
    device: torch.device,
) -> ProbingDataset:
    """Creates the train, validation, and testing datasets using the TrustPilot data

    Args:
        data_path (str): path to dataset (in /data/ folder)
        model_type (str): type of huggingface model to use
        emb_layer (int): what layer of the transformer we want embeddings from. When this is 0, we are using the final output. Otherwise, we will be using the specified (1-indexed) hidden layer output
        device (torch.device): what torch device (cuda or cpu) are we using

    Returns:
        ProbingDataset: the entire parsed dataset
    """
    data = pd.read_csv(data_path)
    data = data.sample(frac=1, random_state=5)
    emb_path = os.path.join(f'{data_path}_embeddings', f'{data_path}_{model_type}_embed{emb_layer}.pt')
    if not os.path.exists(emb_path):
        all_text = data["text"].tolist()
        pooled_embs = create_embeddings(
            all_text, emb_path, model_type, emb_layer, device)
    else:
        pooled_embs = torch.load(emb_path).to(device)

    if "nps" in data_path:
        # Need to convert 4 labels to 1
        ground_truth = torch.tensor(data["label"].replace({10: 0, 20: 1, 30: 2, 40: 3}).tolist(), device=device)
    elif "cola" in data_path:
        ground_truth = torch.tensor(data["label"].tolist(), device=device)
    elif "reddit" in data_path:
        ground_truth = torch.tensor(data["label"].tolist(), device=device)
    elif "fitocracy" in data_path:
        ground_truth = torch.tensor(data["label"].tolist(), device=device)
    elif "facebook" in data_path:
        ground_truth = torch.tensor(data["label"].tolist(), device=device)
    else:
        raise Exception("Dataset does not seem to be for NPS or for CoLA. Please use the correct code to get the right results.")

    orig_strings = data["text"].tolist()
    indices = torch.tensor(range(len(orig_strings)), dtype=torch.long, device=device)
    return ProbingDataset(pooled_embs, ground_truth, indices, orig_strings)

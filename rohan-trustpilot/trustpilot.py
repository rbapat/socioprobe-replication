import os
import json
from dataclasses import dataclass
from datetime import datetime as dt
from typing import Dict, List, Tuple

import tqdm
import torch
import transformers as trfm
from torch.utils.data import Dataset


@dataclass
class Review:
    gender: int  # 0 for male, 1 for female
    age: int  # 0 for young, 1 for old
    review_text: str


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

    def __len__(self):
        return len(self.pooled_embeddings)

    def __getitem__(self, idx):
        return (self.pooled_embeddings[idx], self.ground_truth[idx], self.indices[idx])


def parse_profile(profile_json: Dict) -> List[Review]:
    """Converts the json representation of a profile with reviews to a list of `Review` objects

    Args:
        profile_json (Dict): json of review

    Returns:
        List[Review]: List of reviews with age and gender
    """
    required = ["gender", "reviews", "birth_year"]
    if any(key not in profile_json or profile_json[key] is None for key in required):
        return []

    gender_str = profile_json["gender"].upper()
    if gender_str not in "MF":
        return []

    reviews = []
    gender = 0 if gender_str == "M" else 1
    birth_year = int(profile_json["birth_year"])
    for review in profile_json["reviews"]:
        review_text = "\n".join(review["text"])
        age = dt.fromisoformat(review["date"]).year - birth_year

        if age < 16 or age > 70:
            continue

        if age < 35:
            is_old = 0
        elif age > 45:
            is_old = 1
        else:
            continue

        reviews.append(Review(gender, is_old, review_text))

    return reviews


def load_reviews(path: str) -> List[Review]:
    """Reads the TrustPilot jsonl file from disk and converts it to the reviews

    Args:
        path (str): path to TrustPilot dataset jsonl

    Returns:
        List[Review]: list of `Review` objects created from jsonl
    """
    assert os.path.exists(path) and path.endswith(".jsonl")

    reviews = []
    with open(path) as f:
        for line in f:
            reviews.extend(parse_profile(json.loads(line)))

    return reviews


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

    # print(f"{model_type} has {model_config.num_hidden_layers} hidden layers")
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
    print(
        f"Creating embeddings for {len(all_reviews)} reviews, embedding layer {emb_layer}"
    )

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
    jsonl_path: str,
    model_type: str,
    emb_layer: int,
    device: torch.device,
    use_age: bool = False,
    use_gender: bool = False,
) -> ProbingDataset:
    """Creates the train, validation, and testing datasets using the TrustPilot data

    Args:
        jsonl_path (str): path to TrustPilot jsonl
        model_type (str): type of huggingface model to use
        emb_layer (int): what layer of the transformer we want embeddings from. When this is 0, we are using the final output. Otherwise, we will be using the specified (1-indexed) hidden layer output
        device (torch.device): what torch device (cuda or cpu) are we using
        use_age (bool, optional): Whether or not to make the ground truth age. Must be false if use_gender is true. Defaults to False
        use_gender (bool, optional): Whether or not to make the ground truth gender. Must be false is use_age is true. Defaults to False.

    Returns:
        ProbingDataset: the entire parsed dataset
    """

    assert not use_age or not use_gender

    reviews = load_reviews(jsonl_path)

    if use_age:
        ground_truth = torch.tensor(
            [rev.age for rev in reviews], device=device, dtype=torch.long
        )
    elif use_gender:
        ground_truth = torch.tensor(
            [rev.gender for rev in reviews], device=device, dtype=torch.long
        )
    else:
        raise RuntimeError("use_age and use_gender were both false")

    emb_path = os.path.join(
        "embeddings", model_type, os.path.basename(jsonl_path), f"{emb_layer}.pt"
    )
    if not os.path.exists(emb_path):
        all_text = [rev.review_text for rev in reviews]
        pooled_embeddings = create_embeddings(
            all_text, emb_path, model_type, emb_layer, device
        )
    else:
        pooled_embeddings = torch.load(emb_path).to(device)

    orig_strings = [rev.review_text for rev in reviews]
    indices = torch.tensor(range(len(orig_strings)), dtype=torch.long, device=device)
    return ProbingDataset(pooled_embeddings, ground_truth, indices, orig_strings)

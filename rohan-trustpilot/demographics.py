import os
import random

import torch
import numpy as np
import transformers

import trustpilot


def set_seed(val: int):
    """Sets the seed for all experiments

    Args:
        val (int): seed to set
    """
    transformers.set_seed(val)
    torch.manual_seed(val)
    random.seed(val)
    np.random.seed(val)


def main():
    set_seed(0)

    jsonl_path = os.path.join("data", "united_states.auto-adjusted_gender.jsonl")
    reviews = trustpilot.load_reviews(jsonl_path)

    counts = {"old": 0, "young": 0, "male": 0, "female": 0}
    mapping = ["young", "old", "male", "female"]
    for rev in reviews:
        counts[mapping[rev.age]] += 1
        counts[mapping[rev.gender + 2]] += 1

    total = counts["old"] + counts["young"]
    assert total == counts["male"] + counts["female"]

    pcts = {label: counts[label] / total for label in counts}

    print(f"Total samples: {total}")
    for label in counts:
        print(
            f"There are {counts[label]} {label} samples, which is {pcts[label]*100:.2f}% of the dataset"
        )


if __name__ == "__main__":
    main()

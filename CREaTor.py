#!/usr/bin/env python
import os
import pickle

import click
import torch

from model.model import Model as model_fn


def get_checkpoint_path():
    return os.path.join(os.path.dirname(__file__), "checkpoints", "chk.pth")


def load_model():
    model = model_fn()
    checkpoint_path = get_checkpoint_path()
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    parsed_dict = {}
    for k, v in checkpoint["state_dict"].items():
        if k.startswith("module."):
            k = k[7:]
        parsed_dict[k] = v
    model.load_state_dict(parsed_dict)
    return model.eval()


@click.command()
@click.option("-i", "--pkl", help="input feature(pickle format)")
def main(pkl):
    model = load_model()
    data = pickle.load(open(pkl, "rb"))
    with torch.no_grad():
        output = model(data)
    print(output["expression"][0][0].item())


if __name__ == "__main__":
    main()

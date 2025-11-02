import os
os.environ["KERAS_BACKEND"] = "jax"

import keras
import pathlib
import pandas as pd


import bayesflow as bf

from src.networks import SummaryNet


if __name__ == "__main__":
    print(os.getcwd())

    estimator = keras.saving.load_model("../checkpoints/tflow_complete_pooling_flow_matching_3e4/model.keras")
    print("success")
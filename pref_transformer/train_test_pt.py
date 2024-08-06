import jax
import jax.numpy as jnp
import numpy as np

from bb_data_loading import load_attempt_data, load_BB_data, load_lvl_data
from data_processing import compute_features, to_jnp
from pref_transformer import PT
from train_model import train_model

path_1 = "~/busy-beeway/data/game_data/Experiment_1T5/auto-1a999972a7ab624b/test.2023.03.19.17.29.44/"
path_2 = "~/busy-beeway/data/game_data/Experiment_1T5/auto-1ba807eecf3cf284/test.2022.03.23.12.35.20/"
fill_size = 400
train_split = 0.6
batch_size = 5
n_epochs = 1000

d_1 = load_BB_data(path=path_1)
d_2 = load_BB_data(path=path_2)

f_1 = compute_features(d_1)
f_2 = compute_features(d_2)

if len(f_1) < len(f_2):
    f_2 = f_2[: len(f_1)]
elif len(f_1) > len(f_2):
    f_1 = f_1[: len(f_2)]
else:
    pass

jf_1 = to_jnp(f_1, fill_size=fill_size)
jf_2 = to_jnp(f_2, fill_size=fill_size, labels=("observations_2", "timesteps_2","attn_mask_2"))

data = jf_1 | jf_2 | {"labels": jnp.ones(jf_1["observations"].shape[0])}

t_int = int(jf_1["observations"].shape[0] * train_split)
training_data = {}
test_data = {}
for k in data.keys():
    training_data[k] = data[k][:t_int]
    test_data[k] = data[k][t_int:]

train_model(training_data, test_data, batch_size=batch_size, n_epochs=n_epochs)

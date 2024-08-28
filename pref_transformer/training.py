from functools import partial

from ml_collections import ConfigDict

import jax
import jax.numpy as jnp

import optax
import numpy as np
from flax.training.train_state import TrainState

from jax_utils import next_rng, value_and_multi_grad, mse_loss, cross_ent_loss, kld_loss


class PrefTransformer(object):

    def __init__(self, trans, observation_dim):
        self.trans = trans
        self.observation_dim = observation_dim

        self._train_states = {}

        optimizer_class = optax.adamw
        # May need to reconfigure for our data
        scheduler_class = optax.warmup_cosine_decay_schedule(
            init_value=0,
            peak_value=1e-5,
            warmup_steps=650,
            decay_steps=6500,
            end_value=0,
        )

        tx = optimizer_class(scheduler_class)

        # Reconfigure for our data
        trans_params = self.trans.init(
            {"params": next_rng(), "dropout": next_rng()},
            jnp.zeros((10, 25, self.observation_dim)),
            jnp.ones((10, 25), dtype=jnp.int32),
        )
        self._train_states["trans"] = TrainState.create(
            params=trans_params, tx=tx, apply_fn=None
        )

        model_keys = ["trans"]
        self._model_keys = tuple(model_keys)
        self._total_steps = 0

    def evaluation(self, batch):
        metrics = self._eval_pref_step(self._train_states, next_rng(), batch)
        return metrics

    @partial(jax.jit, static_argnames=("self"))
    def _eval_pref_step(self, train_states, rng, batch):

        def loss_fn(train_params, rng):
            obs_1 = batch["observations"]
            obs_2 = batch["observations_2"]
            timestep_1 = batch["timesteps"]
            timestep_2 = batch["timesteps_2"]
            am_1 = batch["attn_mask"]
            am_2 = batch["attn_mask_2"]
            labels = batch["labels"]

            B, T, _ = batch["observations"].shape

            rng, _ = jax.random.split(rng)

            trans_pred_1, _ = self.trans.apply(
                train_params["trans"],
                obs_1,
                timestep_1,
                training=False,
                attn_mask=am_1,
                rngs={"dropout": rng},
            )
            trans_pred_2, _ = self.trans.apply(
                train_params["trans"],
                obs_2,
                timestep_2,
                training=False,
                attn_mask=am_2,
                rngs={"dropout": rng},
            )

            trans_pred_1 = trans_pred_1["weighted_sum"]
            trans_pred_2 = trans_pred_2["weighted_sum"]

            sum_pred_1 = jnp.mean(trans_pred_1.reshape(B, T), axis=1).reshape(-1, 1)
            sum_pred_2 = jnp.mean(trans_pred_2.reshape(B, T), axis=1).reshape(-1, 1)

            logits = jnp.concatenate([sum_pred_1, sum_pred_2], axis=1)

            return cross_ent_loss(logits, label_target)

        train_params = {key: train_states[key].params for key in self.model_keys}

        return {"eval_trans_loss": loss_fn(train_params,rng)}

    def train(self, batch):
        self._total_steps += 1
        self._train_states, metrics = self._train_pref_step(
            self._train_states, next_rng(), batch
        )
        return metrics

    @partial(jax.jit, static_argnames=("self"))
    def _train_pref_step(self, train_states, rng, batch):

        def loss_fn(train_params, rng):
            obs_1 = batch["observations"]
            obs_2 = batch["observations_2"]
            timestep_1 = batch["timesteps"]
            timestep_2 = batch["timesteps_2"]
            am_1 = batch["attn_mask"]
            am_2 = batch["attn_mask_2"]
            labels = batch["labels"]

            B, T, _ = batch["observations"].shape

            rng, _ = jax.random.split(rng)

            trans_pred_1, _ = self.trans.apply(
                train_params["trans"],
                obs_1,
                timestep_1,
                training=True,
                attn_mask=am_1,
                rngs={"dropout": rng},
            )
            trans_pred_2, _ = self.trans.apply(
                train_params["trans"],
                obs_2,
                timestep_2,
                training=True,
                attn_mask=am_2,
                rngs={"dropout": rng},
            )

            trans_pred_1 = trans_pred_1["weighted_sum"]
            trans_pred_2 = trans_pred_2["weighted_sum"]

            sum_pred_1 = jnp.mean(trans_pred_1.reshape(B, T), axis=1).reshape(-1, 1)
            sum_pred_2 = jnp.mean(trans_pred_2.reshape(B, T), axis=1).reshape(-1, 1)

            logits = jnp.concatenate([sum_pred_1, sum_pred_2], axis=1)

            loss_collection = {}

            rng, split_rng = jax.random.split(rng)

            """ reward function loss """
            label_target = jax.lax.stop_gradient(labels)
            trans_loss = cross_ent_loss(logits, label_target)
            loss_collection["trans"] = trans_loss
            return tuple(loss_collection[key] for key in self.model_keys), locals()

        train_params = {key: train_states[key].params for key in self.model_keys}
        (_, aux_values), grads = value_and_multi_grad(
            loss_fn, len(self.model_keys), has_aux=True
        )(train_params, rng)

        new_train_states = {
            key: train_states[key].apply_gradients(grads=grads[i][key])
            for i, key in enumerate(self.model_keys)
        }

        metrics = dict(
            trans_loss=aux_values["trans_loss"],
        )

        return new_train_states, metrics

    @property
    def model_keys(self):
        return self._model_keys

    @property
    def train_states(self):
        return self._train_states

    @property
    def train_params(self):
        return {key: self.train_states[key].params for key in self.model_keys}

    @property
    def total_steps(self):
        return self._total_steps


class intervention_MLP(object):

    def __init__(self, imlp, observation_dim):
        self.imlp = imlp
        self.observation_dim = observation_dim

        self._train_states = {}

        optimizer_class = optax.adamw
        # May need to reconfigure for our data
        scheduler_class = optax.warmup_cosine_decay_schedule(
            init_value=0,
            peak_value=1e-4,
            warmup_steps=650,
            decay_steps=6500,
            end_value=0,
        )

        tx = optimizer_class(scheduler_class)

        # Reconfigure for our data
        imlp_params = self.imlp.init(
            {"params": next_rng(), "dropout": next_rng()},
            jnp.zeros((10, self.observation_dim)),
        )
        self._train_states["imlp"] = TrainState.create(
            params=imlp_params, tx=tx, apply_fn=None
        )

        model_keys = ["imlp_loss,imlp_accuracy"]
        self._model_keys = tuple(model_keys)
        self._total_steps = 0

    def evaluation(self, batch):
        metrics = self._eval_step(self._train_states, next_rng(), batch)
        return metrics

    @partial(jax.jit, static_argnames=("self"))
    def _eval_step(self, train_states, rng, batch):

        def loss_fn(train_params, rng):
            obs = batch["observations"]
            labels = batch["labels"]

            rng, _ = jax.random.split(rng)

            imlp_pred = self.imlp.apply(
                train_params["imlp"],
                obs,
                training=False,
                rngs={"dropout": rng},
            )

            rng, split_rng = jax.random.split(rng)

            """ reward function loss """
            pred_labels = (imlp_pred > 0).astype(jnp.float32)
            label_target = jax.lax.stop_gradient(labels)
            imlp_loss = optax.sigmoid_binary_cross_entropy(imlp_pred, label_target).mean()
            imlp_acc = (pred_labels == labels).mean()
            loss_collection["imlp_loss"] = imlp_loss
            loss_collection["imlp_accuracy"] = imlp_acc
            return imlp_loss, imlp_acc

        train_params = {key: train_states[key].params for key in self.model_keys}
        (_, aux_values), _ = value_and_multi_grad(
            loss_fn, len(self.model_keys), has_aux=True
        )(train_params, rng)

        metrics = dict(
            eval_trans_loss=aux_values["trans_loss"],
        )

        return metrics

    def train(self, batch):
        self._total_steps += 1
        self._train_states, metrics = self._train_pref_step(
            self._train_states, next_rng(), batch
        )
        return metrics

    @partial(jax.jit, static_argnames=("self"))
    def _train_pref_step(self, train_states, rng, batch):

        def loss_fn(train_params, rng):
            obs_1 = batch["observations"]
            obs_2 = batch["observations_2"]
            timestep_1 = batch["timesteps"]
            timestep_2 = batch["timesteps_2"]
            am_1 = batch["attn_mask"]
            am_2 = batch["attn_mask_2"]
            labels = batch["labels"]

            B, T, _ = batch["observations"].shape

            rng, _ = jax.random.split(rng)

            trans_pred_1, _ = self.trans.apply(
                train_params["trans"],
                obs_1,
                timestep_1,
                training=True,
                attn_mask=am_1,
                rngs={"dropout": rng},
            )
            trans_pred_2, _ = self.trans.apply(
                train_params["trans"],
                obs_2,
                timestep_2,
                training=True,
                attn_mask=am_2,
                rngs={"dropout": rng},
            )

            trans_pred_1 = trans_pred_1["weighted_sum"]
            trans_pred_2 = trans_pred_2["weighted_sum"]

            sum_pred_1 = jnp.mean(trans_pred_1.reshape(B, T), axis=1).reshape(-1, 1)
            sum_pred_2 = jnp.mean(trans_pred_2.reshape(B, T), axis=1).reshape(-1, 1)

            logits = jnp.concatenate([sum_pred_1, sum_pred_2], axis=1)

            loss_collection = {}

            rng, split_rng = jax.random.split(rng)

            """ reward function loss """
            label_target = jax.lax.stop_gradient(labels)
            trans_loss = cross_ent_loss(logits, label_target)
            loss_collection["trans"] = trans_loss
            return tuple(loss_collection[key] for key in self.model_keys), locals()

        train_params = {key: train_states[key].params for key in self.model_keys}
        (_, aux_values), grads = value_and_multi_grad(
            loss_fn, len(self.model_keys), has_aux=True
        )(train_params, rng)

        new_train_states = {
            key: train_states[key].apply_gradients(grads=grads[i][key])
            for i, key in enumerate(self.model_keys)
        }

        metrics = dict(
            trans_loss=aux_values["trans_loss"],
        )

        return new_train_states, metrics


    @property
    def model_keys(self):
        return self._model_keys

    @property
    def train_states(self):
        return self._train_states

    @property
    def train_params(self):
        return {key: self.train_states[key].params for key in self.model_keys}

    @property
    def total_steps(self):
        return self._total_steps
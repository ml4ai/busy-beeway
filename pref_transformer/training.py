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

    def get_reward(self, batch):
        return self._get_reward_step(self._train_states, batch)

    @partial(jax.jit, static_argnames=("self"))
    def _get_reward_step(self, train_states, batch):
        obs = batch["observations"]
        timestep = batch["timesteps"]
        # n_obs = batch['next_observations']
        attn_mask = batch["attn_mask"]

        train_params = {key: train_states[key].params for key in self.model_keys}
        trans_pred, attn_weights = self.trans.apply(
            train_params["trans"], obs, timestep, attn_mask=attn_mask
        )
        return trans_pred["value"], attn_weights[-1]

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

            loss_collection = {}

            rng, split_rng = jax.random.split(rng)

            """ reward function loss """
            label_target = jax.lax.stop_gradient(labels)
            trans_loss = cross_ent_loss(logits, label_target)
            loss_collection["trans"] = trans_loss
            return tuple(loss_collection[key] for key in self.model_keys), locals()

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

    def train_semi(self, labeled_batch, unlabeled_batch, lmd, tau):
        self._total_steps += 1
        self._train_states, metrics = self._train_semi_pref_step(
            self._train_states, labeled_batch, unlabeled_batch, lmd, tau, next_rng()
        )
        return metrics

    @partial(jax.jit, static_argnames=("self"))
    def _train_semi_pref_step(
        self, train_states, labeled_batch, unlabeled_batch, lmd, tau, rng
    ):
        def compute_logits(train_params, batch, rng):
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
            return logits, labels

        def loss_fn(train_params, lmd, tau, rng):
            rng, _ = jax.random.split(rng)
            logits, labels = compute_logits(train_params, labeled_batch, rng)
            u_logits, _ = compute_logits(train_params, unlabeled_batch, rng)

            loss_collection = {}

            rng, split_rng = jax.random.split(rng)

            """ reward function loss """
            label_target = jax.lax.stop_gradient(labels)
            trans_loss = cross_ent_loss(logits, label_target)

            u_confidence = jnp.max(jax.nn.softmax(u_logits, axis=-1), axis=-1)
            pseudo_labels = jnp.argmax(u_logits, axis=-1)
            pseudo_label_target = jax.lax.stop_gradient(pseudo_labels)

            loss_ = optax.softmax_cross_entropy(
                logits=u_logits,
                labels=jax.nn.one_hot(pseudo_label_target, num_classes=2),
            )
            u_trans_loss = jnp.sum(jnp.where(u_confidence > tau, loss_, 0)) / (
                jnp.count_nonzero(u_confidence > tau) + 1e-4
            )
            u_trans_ratio = (
                jnp.count_nonzero(u_confidence > tau) / len(u_confidence) * 100
            )

            # labeling neutral cases.
            binarized_idx = jnp.where(unlabeled_batch["labels"][:, 0] != 0.5, 1.0, 0.0)
            real_label = jnp.argmax(unlabeled_batch["labels"], axis=-1)
            u_trans_acc = (
                jnp.sum(
                    jnp.where(pseudo_label_target == real_label, 1.0, 0.0)
                    * binarized_idx
                )
                / jnp.sum(binarized_idx)
                * 100
            )

            loss_collection["trans"] = last_loss = trans_loss + lmd * u_trans_loss
            return tuple(loss_collection[key] for key in self.model_keys), locals()

        train_params = {key: train_states[key].params for key in self.model_keys}
        (_, aux_values), grads = value_and_multi_grad(
            loss_fn, len(self.model_keys), has_aux=True
        )(train_params, lmd, tau, rng)

        new_train_states = {
            key: train_states[key].apply_gradients(grads=grads[i][key])
            for i, key in enumerate(self.model_keys)
        }

        metrics = dict(
            trans_loss=aux_values["trans_loss"],
            u_trans_loss=aux_values["u_trans_loss"],
            last_loss=aux_values["last_loss"],
            u_trans_ratio=aux_values["u_trans_ratio"],
            u_train_acc=aux_values["u_trans_acc"],
        )

        return new_train_states, metrics

    def train_regression(self, batch):
        self._total_steps += 1
        self._train_states, metrics = self._train_regression_step(
            self._train_states, next_rng(), batch
        )
        return metrics

    @partial(jax.jit, static_argnames=("self"))
    def _train_regression_step(self, train_states, rng, batch):

        def loss_fn(train_params, rng):
            observations = batch["observations"]
            next_observations = batch["next_observations"]
            rewards = batch["rewards"]

            in_obs = jnp.concatenate([observations, next_observations], axis=-1)

            loss_collection = {}

            rng, split_rng = jax.random.split(rng)

            """ reward function loss """
            rf_pred = self.rf.apply(train_params["rf"], observations)
            reward_target = jax.lax.stop_gradient(rewards)
            rf_loss = mse_loss(rf_pred, reward_target)

            loss_collection["rf"] = rf_loss
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
            rf_loss=aux_values["rf_loss"],
            average_rf=aux_values["rf_pred"].mean(),
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

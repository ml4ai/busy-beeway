from training import PrefTransformer
from pref_transformer import PT
import numpy as np
from utils import Timer, index_batch, prefix_metrics, save_pickle, set_random_seed
from jax_utils import batch_to_jax
from flax.training.early_stopping import EarlyStopping
from logging_utils import logger, setup_logger
import os.path as osp


def train_model(
    training_data,
    test_data,
    batch_size=256,
    n_epochs=76000,
    eval_period=5,
    do_early_stop=False,
    criteria_key="reward/eval_trans_loss",
    seed=2024,
    save_dir="~/busy-beeway/pref_transformer/logs",
    save_model=True,
):

    save_dir = osp.expanduser(save_dir)
    setup_logger(
        variant=None, seed=seed, base_log_dir=save_dir, include_exp_prefix_sub_dir=False
    )
    set_random_seed(seed)
    rng = np.random.default_rng(seed)
    data_size, query_len, observation_dim = training_data["observations"].shape
    eval_data_size = test_data["observations"].shape[0]
    trans = PT()
    reward_model = PrefTransformer(trans, observation_dim)
    interval = int(data_size / batch_size) + 1
    eval_interval = int(eval_data_size / batch_size) + 1
    early_stop = EarlyStopping(min_delta=1e-3, patience=10)
    c_best_epoch = np.nan
    c_criteria_key = np.nan
    for epoch in range(n_epochs + 1):
        metrics = {
            "epoch": epoch,
            "train_time": np.nan,
            "reward/trans_loss": [],
            "reward/eval_trans_loss": [],
            "best_epoch": c_best_epoch,
            f"{criteria_key}_best": c_criteria_key,
        }
        if epoch:
            # train phase
            shuffled_idx = rng.permutation(data_size)
            for i in range(interval):
                start_pt = i * batch_size
                end_pt = min((i + 1) * batch_size, data_size)
                with Timer() as train_timer:
                    # train
                    batch = batch_to_jax(
                        index_batch(training_data, shuffled_idx[start_pt:end_pt])
                    )
                    for key, val in prefix_metrics(
                        reward_model.train(batch), "reward"
                    ).items():
                        metrics[key].append(val)
            metrics["train_time"] = train_timer()
        else:
            # for using early stopping with train loss.
            metrics["reward/trans_loss"] = np.nan

        # eval phase
        if epoch % eval_period == 0:
            for j in range(eval_interval):
                eval_start_pt, eval_end_pt = j * batch_size, min(
                    (j + 1) * batch_size, eval_data_size
                )
                # batch_eval = batch_to_jax(index_batch(pref_eval_dataset, range(eval_start_pt, eval_end_pt)))
                batch_eval = batch_to_jax(
                    index_batch(test_data, list(range(eval_start_pt, eval_end_pt)))
                )
                for key, val in prefix_metrics(
                    reward_model.evaluation(batch_eval), "reward"
                ).items():
                    metrics[key].append(val)
            criteria = np.mean(metrics[criteria_key])
            early_stop = early_stop.update(criteria)
            if early_stop.should_stop and do_early_stop:
                for key, val in metrics.items():
                    if isinstance(val, list):
                        if len(val):
                            metrics[key] = np.mean(val)
                        else:
                            metrics[key] = np.nan
                logger.record_dict(metrics)
                logger.dump_tabular(with_prefix=False, with_timestamp=False)
                print("Met early stopping criteria, breaking...")
                break
            elif epoch > 0 and early_stop.has_improved:
                c_best_epoch = epoch
                c_criteria_key = criteria
                metrics["best_epoch"] = c_best_epoch
                metrics[f"{criteria_key}_best"] = c_criteria_key
                save_data = {"reward_model": reward_model, "epoch": epoch}
                save_pickle(save_data, "best_model.pkl", save_dir)

        for key, val in metrics.items():
            if isinstance(val, list):
                if len(val):
                    metrics[key] = np.mean(val)
                else:
                    metrics[key] = np.nan
        logger.record_dict(metrics)
        logger.dump_tabular(with_prefix=False, with_timestamp=False)
    if save_model:
        save_data = {"reward_model": reward_model, "epoch": epoch}
        save_pickle(save_data, "model.pkl", save_dir)

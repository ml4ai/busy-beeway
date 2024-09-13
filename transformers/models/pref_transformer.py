from flax import linen as nn
import jax.numpy as jnp
from jax import lax
import transformers.models.ops as ops


class GPT2MLP(nn.Module):
    embd_dim: int = 64
    intermediate_dim: int = 256
    resid_dropout: float = 0.1

    @nn.compact
    def __call__(self, x, training=False):
        x = nn.Dense(features=self.intermediate_dim, dtype=jnp.bfloat16)(x)
        x = ops.apply_activation(x, activation="relu")
        x = nn.Dense(features=self.embd_dim, dtype=jnp.bfloat16)(x)
        x = nn.Dropout(rate=self.resid_dropout)(x, deterministic=not training)
        return x


class GPT2SelfAttention(nn.Module):
    embd_dim: int = 64
    num_heads: int = 4
    attn_dropout: float = 0.1
    resid_dropout: float = 0.1
    max_pos: int = 1024

    @nn.compact
    def __call__(self, x, attn_mask=None, training=False):
        head_dim = self.embd_dim // self.num_heads
        x = nn.Dense(features=3 * self.embd_dim, dtype=jnp.bfloat16)(x)

        query, key, value = jnp.split(x, 3, axis=2)

        query = ops.split_heads(query, self.num_heads, head_dim)
        value = ops.split_heads(value, self.num_heads, head_dim)
        key = ops.split_heads(key, self.num_heads, head_dim)

        query_len, key_len = query.shape[-2], key.shape[-2]
        casual_mask = jnp.tril(jnp.ones((1, 1, self.max_pos, self.max_pos)))[
            :, :, key_len - query_len : key_len, :key_len
        ]
        casual_mask = casual_mask.astype(bool)

        attn_dropout = nn.Dropout(rate=self.attn_dropout)
        out, _attn_weights = ops.attention(
            query,
            key,
            value,
            casual_mask,
            -1e4,
            attn_dropout,
            True,
            training,
            attn_mask,
        )
        out = ops.merge_heads(out, self.num_heads, head_dim)

        out = nn.Dense(features=self.embd_dim, dtype=jnp.bfloat16)(out)

        out = nn.Dropout(rate=self.resid_dropout)(out, deterministic=not training)
        return out, _attn_weights


class GPT2Block(nn.Module):
    embd_dim: int = 64
    num_heads: int = 4
    attn_dropout: float = 0.1
    resid_dropout: float = 0.1
    intermediate_dim: int = 256
    max_pos: int = 1024
    eps: float = 1e-05

    @nn.compact
    def __call__(self, x, attn_mask=None, training=False):
        residual = x
        x = nn.LayerNorm(epsilon=self.eps, dtype=jnp.bfloat16)(x)
        x, _attn_weights = GPT2SelfAttention(
            embd_dim=self.embd_dim,
            num_heads=self.num_heads,
            attn_dropout=self.attn_dropout,
            resid_dropout=self.resid_dropout,
            max_pos=self.max_pos,
        )(x, attn_mask, training)
        x += residual
        residual = x
        x = nn.LayerNorm(epsilon=self.eps, dtype=jnp.bfloat16)(x)
        x = GPT2MLP(
            embd_dim=self.embd_dim,
            intermediate_dim=self.intermediate_dim,
            resid_dropout=self.resid_dropout,
        )(x, training)
        x += residual
        return x, _attn_weights


class GPT2Model(nn.Module):
    embd_dim: int = 64
    num_heads: int = 4
    attn_dropout: float = 0.1
    resid_dropout: float = 0.1
    intermediate_dim: int = 256
    num_layers: int = 1
    embd_dropout: float = 0.1
    max_pos: int = 1024
    eps: float = 1e-05

    @nn.compact
    def __call__(self, input_embds, attn_mask=None, training=False):
        input_shape = input_embds.shape[:-1]
        batch_size = input_embds.shape[0]

        if attn_mask is not None:
            attn_mask = ops.get_attention_mask(attn_mask, batch_size)

        x = input_embds

        x = nn.Dropout(rate=self.embd_dropout)(x, deterministic=not training)
        output_shape = input_shape + (x.shape[-1],)

        attn_weights_list = []

        def blocks(i, y):
            (
                attn_mask,
                training,
                embd_dim,
                num_heads,
                attn_dropout,
                intermediate_dim,
                max_pos,
                eps,
                x,
                attn_weights_list,
            ) = y
            x, attn_weights = GPT2Block(
                embd_dim=embd_dim,
                num_heads=num_heads,
                attn_dropout=attn_dropout,
                intermediate_dim=intermediate_dim,
                max_pos=max_pos,
                eps=eps,
            )(x, attn_mask, training)

            attn_weights_list.append(attn_weights)
            return (
                attn_mask,
                training,
                embd_dim,
                num_heads,
                attn_dropout,
                intermediate_dim,
                max_pos,
                eps,
                x,
                attn_weights_list,
            )

        y = (
            attn_mask,
            training,
            self.embd_dim,
            self.num_heads,
            self.attn_dropout,
            self.intermediate_dim,
            self.max_pos,
            self.eps,
            x,
            attn_weights_list,
        )
        (
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            x,
            attn_weights_list,
        ) = lax.fori_loop(0, self.num_layers, blocks, y, unroll=True)
        # for i in range(self.num_layers):
        #     kwargs = {
        #         "layer_past": past_key_values[i],
        #         "attn_mask": attn_mask,
        #         "head_mask": head_mask[i],
        #         "training": training,
        #     }
        #     x, present, attn_weights = GPT2Block(
        #         embd_dim=self.embd_dim,
        #         num_heads=self.num_heads,
        #         attn_dropout=self.attn_dropout,
        #         intermediate_dim=self.intermediate_dim,
        #         activation=self.activation,
        #         max_pos=self.max_pos,
        #         eps=self.eps,
        #     )(x, **kwargs)

        #     attn_weights_list.append(attn_weights)
        x = nn.LayerNorm(epsilon=self.eps, dtype=jnp.bfloat16)(x)
        return {
            "last_hidden_state": x,
            "attn_weights_list": attn_weights_list,
        }


class PT(nn.Module):
    observation_dim: int = 327
    max_episode_steps: int = 1219
    embd_dim: int = 64
    pref_attn_embd_dim: int = 64
    num_heads: int = 4
    attn_dropout: float = 0.1
    resid_dropout: float = 0.1
    intermediate_dim: int = 256
    num_layers: int = 1
    embd_dropout: float = 0.1
    max_pos: int = 1024
    eps: float = 1e-05

    @nn.compact
    def __call__(self, states, timesteps, training=False, attn_mask=None):
        batch_size, seq_length = states.shape[0], states.shape[1]
        if attn_mask is None:
            attn_mask = jnp.ones((batch_size, seq_length), dtype=jnp.float32)

        embd_states = nn.Dense(features=self.embd_dim, dtype=jnp.bfloat16)(states)
        embd_timesteps = nn.Embed(
            num_embeddings=self.max_episode_steps + 1,
            features=self.embd_dim,
            dtype=jnp.bfloat16,
        )(timesteps)

        embd_states = embd_states + embd_timesteps

        inputs = nn.LayerNorm(epsilon=self.eps, dtype=jnp.bfloat16)(embd_states)

        transformer_outputs = GPT2Model(
            embd_dim=self.embd_dim,
            num_heads=self.num_heads,
            attn_dropout=self.attn_dropout,
            resid_dropout=self.resid_dropout,
            intermediate_dim=self.intermediate_dim,
            num_layers=self.num_layers,
            embd_dropout=self.embd_dropout,
            max_pos=self.max_pos,
            eps=self.eps,
        )(input_embds=inputs, attn_mask=attn_mask, training=training)

        hidden_output = transformer_outputs["last_hidden_state"]
        attn_weights_list = transformer_outputs["attn_weights_list"]

        x = nn.Dense(features=2 * self.pref_attn_embd_dim + 1, dtype=jnp.bfloat16)(
            hidden_output
        )

        num_heads = 1

        query, key, value = jnp.split(
            x, [self.pref_attn_embd_dim, self.pref_attn_embd_dim * 2], axis=2
        )
        query = ops.split_heads(query, num_heads, self.pref_attn_embd_dim)
        key = ops.split_heads(key, num_heads, self.pref_attn_embd_dim)
        value = ops.split_heads(value, num_heads, 1)

        query_len, key_len = query.shape[-2], key.shape[-2]
        casual_mask = jnp.ones((1, 1, seq_length, seq_length))[
            :, :, key_len - query_len : key_len, :key_len
        ]
        casual_mask = casual_mask.astype(bool)

        attn_dropout = nn.Dropout(rate=0.0)
        new_attn_mask = ops.get_attention_mask(attn_mask, batch_size)
        out, last_attn_weights = ops.attention(
            query,
            key,
            value,
            casual_mask,
            -1e-4,
            attn_dropout,
            scale_attn_weights=True,
            training=training,
            attn_mask=new_attn_mask,
        )
        attn_weights_list.append(last_attn_weights)

        output = ops.merge_heads(out, num_heads, 1)

        return {"weighted_sum": output, "value": value}, attn_weights_list

from flax import linen as nn
import jax.numpy as jnp
import transformers.models.ops


class GPT2MLP(nn.Module):
    embd_dim: int = 64
    intermediate_dim: int = 256
    activation: str = "relu"
    resid_dropout: float = 0.1

    @nn.compact
    def __call__(self, x, training=False):
        x = nn.Dense(features=self.intermediate_dim)(x)
        x = ops.apply_activation(x, activation=self.activation)
        x = nn.Dense(features=self.embd_dim)(x)
        x = nn.Dropout(rate=self.resid_dropout)(x, deterministic=not training)
        return x


class GPT2SelfAttention(nn.Module):
    embd_dim: int = 64
    num_heads: int = 4
    attn_dropout: float = 0.1
    resid_dropout: float = 0.1
    max_pos: int = 1024

    @nn.compact
    def __call__(
        self, x, layer_past=None, attn_mask=None, head_mask=None, training=False
    ):
        head_dim = self.embd_dim // self.num_heads
        x = nn.Dense(features=3 * self.embd_dim)(x)

        query, key, value = jnp.split(x, 3, axis=2)

        query = ops.split_heads(query, self.num_heads, head_dim)
        value = ops.split_heads(value, self.num_heads, head_dim)
        key = ops.split_heads(key, self.num_heads, head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = jnp.concatenate((past_key, key), axis=-2)
            value = jnp.concatenate((past_value, value), axis=-2)

        present = None
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
            head_mask,
        )
        out = ops.merge_heads(out, self.num_heads, head_dim)

        out = nn.Dense(features=self.embd_dim)(out)

        out = nn.Dropout(rate=self.resid_dropout)(out, deterministic=not training)
        return out, present, _attn_weights


class GPT2Block(nn.Module):
    embd_dim: int = 64
    num_heads: int = 4
    attn_dropout: float = 0.1
    resid_dropout: float = 0.1
    intermediate_dim: int = 256
    max_pos: int = 1024
    activation: str = "relu"
    eps: float = 1e-05

    @nn.compact
    def __call__(
        self, x, layer_past=None, attn_mask=None, head_mask=None, training=False
    ):
        residual = x
        x = nn.LayerNorm(epsilon=self.eps)(x)
        kwargs = {
            "layer_past": layer_past,
            "attn_mask": attn_mask,
            "head_mask": head_mask,
            "training": training,
        }
        x, present, _attn_weights = GPT2SelfAttention(
            embd_dim=self.embd_dim,
            num_heads=self.num_heads,
            attn_dropout=self.attn_dropout,
            resid_dropout=self.resid_dropout,
            max_pos=self.max_pos,
        )(x, **kwargs)
        x += residual
        residual = x
        x = nn.LayerNorm(epsilon=self.eps)(x)
        x = GPT2MLP(
            embd_dim=self.embd_dim,
            intermediate_dim=self.intermediate_dim,
            resid_dropout=self.resid_dropout,
            activation=self.activation,
        )(x, training)
        x += residual
        return x, present, _attn_weights


class GPT2Model(nn.Module):
    embd_dim: int = 64
    num_heads: int = 4
    attn_dropout: float = 0.1
    resid_dropout: float = 0.1
    intermediate_dim: int = 256
    activation: str = "relu"
    num_layers: int = 1
    embd_dropout: float = 0.1
    max_pos: int = 1024
    eps: float = 1e-05

    @nn.compact
    def __call__(self, input_embds, attn_mask=None, training=False):
        input_shape = input_embds.shape[:-1]
        batch_size = input_embds.shape[0]
        past_length = 0
        past_key_values = tuple([None] * self.num_layers)
        position_ids = jnp.arange(start=past_length, stop=input_shape[-1] + past_length)
        position_ids = jnp.reshape(
            jnp.expand_dims(position_ids, axis=0), newshape=(-1, input_shape[-1])
        )
        if attn_mask is not None:
            attn_mask = ops.get_attention_mask(attn_mask, batch_size)

        head_mask = [None] * self.num_layers

        x = input_embds

        x = nn.Dropout(rate=self.embd_dropout)(x, deterministic=not training)
        output_shape = input_shape + (x.shape[-1],)

        presents = None
        attn_weights_list = []
        for i in range(self.num_layers):
            kwargs = {
                "layer_past": past_key_values[i],
                "attn_mask": attn_mask,
                "head_mask": head_mask[i],
                "training": training,
            }
            x, present, attn_weights = GPT2Block(
                embd_dim=self.embd_dim,
                num_heads=self.num_heads,
                attn_dropout=self.attn_dropout,
                intermediate_dim=self.intermediate_dim,
                activation=self.activation,
                max_pos=self.max_pos,
                eps=self.eps,
            )(x, **kwargs)

            attn_weights_list.append(attn_weights)
        x = nn.LayerNorm(epsilon=self.eps)(x)
        return {
            "last_hidden_state": x,
            "past_key_values": presents,
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
    activation: str = "relu"
    num_layers: int = 1
    embd_dropout: float = 0.1
    max_pos: int = 1024
    eps: float = 1e-05

    @nn.compact
    def __call__(self, states, timesteps, training=False, attn_mask=None):
        batch_size, seq_length = states.shape[0], states.shape[1]
        if attn_mask is None:
            attn_mask = jnp.ones((batch_size, seq_length), dtype=jnp.float32)

        embd_states = nn.Dense(features=self.embd_dim)(states)
        embd_timesteps = nn.Embed(
            num_embeddings=self.max_episode_steps + 1, features=self.embd_dim
        )(timesteps)

        embd_states = embd_states + embd_timesteps

        inputs = nn.LayerNorm(epsilon=self.eps)(embd_states)

        transformer_outputs = GPT2Model(
            embd_dim=self.embd_dim,
            num_heads=self.num_heads,
            attn_dropout=self.attn_dropout,
            resid_dropout=self.resid_dropout,
            intermediate_dim=self.intermediate_dim,
            activation=self.activation,
            num_layers=self.num_layers,
            embd_dropout=self.embd_dropout,
            max_pos=self.max_pos,
            eps=self.eps,
        )(input_embds=inputs, attn_mask=attn_mask, training=training)

        hidden_output = transformer_outputs["last_hidden_state"]
        attn_weights_list = transformer_outputs["attn_weights_list"]

        x = nn.Dense(features=2 * self.pref_attn_embd_dim + 1)(hidden_output)

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
            head_mask=None,
        )
        attn_weights_list.append(last_attn_weights)

        output = ops.merge_heads(out, num_heads, 1)

        return {"weighted_sum": output, "value": value}, attn_weights_list
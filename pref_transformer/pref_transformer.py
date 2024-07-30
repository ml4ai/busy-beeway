from flax import linen as nn
import jax.numpy as jnp
import ops

class GPT2MLP(nn.Module):
    @nn.compact
    def __call__(self,x,training=False):
        x = nn.Dense(features=128)(x)
        x = ops.apply_activation(x, activation="relu")
        x = nn.Dense(features=256)(x)
        x = nn.Dropout(rate=0.1)(x, deterministic=not training)
        return x

class GPT2SelfAttention(nn.Module):
    @nn.compact
    def __call__(self,
                 x,
                 layer_past=None,
                 attn_mask=None,
                 head_mask=None,
                 training=False):

        x = nn.Dense(features=3*256)(x)

        query, key, value = jnp.split(x,3,axis=2)

        query = ops.split_heads(query, 1, 256)
        value = ops.split_heads(value, 1, 256)
        key = ops.split_heads(key, 1, 256)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = jnp.concatenate((past_key,key),axis=-2)
            value = jnp.concatenate((past_value,value),axis=-2)

        present = None
        query_len, key_len = query.shape[-2], key.shape[-2]
        casual_mask = jnp.tril(jnp.ones((1, 1, 1024, 1024)))[:, :, key_len - query_len :key_len, :key_len]
        casual_mask = casual_mask.astype(bool)

        attn_dropout = nn.Dropout(rate=0.1)
        out, _attn_weights = ops.attention(query,
                                           key,
                                           value,
                                           casual_mask,
                                           -1e4,
                                           attn_dropout,
                                           True,
                                           training,
                                           attn_mask,
                                           head_mask)
        out = ops.merge_heads(out, 1, 256)

        out = nn.Dense(features=256)(out)

        out = nn.Dropout(rate=0.1)(out, deterministic=not training)
        return out, present, _attn_weights

class GPT2Block(nn.Module):
    @nn.compact
    def __call__(self,
                 x,
                 layer_past=None,
                 attn_mask=None,
                 head_mask=None,
                 training=False):
        residual = x
        x = nn.LayerNorm(epsilon=1e-05)(x)
        kwargs = {'layer_past': layer_past, 'attn_mask': attn_mask,
                  'head_mask': head_mask, 'training': training}
        x, present, _attn_weights = GPT2SelfAttention()(x, **kwargs)
        x += residual
        residual = x
        x = nn.LayerNorm(epsilon=1e-05)(x)
        x = GPT2MLP()(x,training)
        x += residual
        return x, present, _attn_weights

class GPT2Model(nn.Module):
    @nn.compact
    def __call__(self,
                 input_embds,
                 attn_mask=None,
                 training=False):
        input_shape = input_embds.shape[:-1]
        batch_size = input_embds.shape[0]
        past_length = 0
        past_key_values = tuple([None] * 3)
        position_ids = jnp.arange(start=past_length, stop=input_shape[-1] + past_length)
        position_ids = jnp.reshape(jnp.expand_dims(position_ids, axis=0),
                                   newshape=(-1, input_shape[-1]))
        if attn_mask is not None:
            attn_mask = ops.get_attention_mask(attn_mask, batch_size)

        head_mask = [None] * 3

        x = input_embds

        x = nn.Dropout(rate=0.1)(x, deterministic=not training)
        output_shape = input_shape + (x.shape[-1],)

        presents = None
        attn_weights_list = []
        for i in range(3):
            kwargs = {'layer_past': past_key_values[i], 'attn_mask': attn_mask,
                      'head_mask': head_mask[i], 'training': training}
            x, present, attn_weights = GPT2Block()(x, **kwargs)

            attn_weights_list.append(attn_weights)
        x = nn.LayerNorm(epsilon=1e-05)(x)
        return {'last_hidden_state': x, 'past_key_values': presents, 'attn_weights_list': attn_weights_list}

class PT(nn.Module):
    @nn.compact
    def __call__(self,
                 states,
                 timesteps,
                 training=False,
                 attn_mask=None):
        batch_size, seq_length = states.shape[0], states.shape[1]
        if attn_mask is None:
            attn_mask = jnp.ones((batch_size, seq_length), dtype=jnp.float32)

        embd_states = nn.Dense(features=256)(states)
        embd_timesteps = nn.Embed(num_embeddings=1001,features=256)(timesteps)

        embd_states = embd_states + embd_timesteps

        inputs = nn.LayerNorm(epsilon=1e-05)(embd_states)

        transformer_outputs = GPT2Model()(input_embds=inputs,
                                          attn_mask=attn_mask,
                                          training=training)

        hidden_output = transformer_outputs["last_hidden_state"]
        attn_weights_list = transformer_outputs["attn_weights_list"]

        x = nn.Dense(features=2 * 256 + 1)(hidden_output)

        num_heads = 1

        query, key, value = jnp.split(x, [256, 256 * 2], axis=2)
        query = ops.split_heads(query, num_heads, 256)
        key = ops.split_heads(key, num_heads, 256)
        value = ops.split_heads(value, num_heads, 1)

        query_len, key_len = query.shape[-2], key.shape[-2]
        casual_mask = jnp.ones((1, 1, seq_length, seq_length))[:, :, key_len - query_len :key_len, :key_len]
        casual_mask = casual_mask.astype(bool)

        attn_dropout = nn.Dropout(rate=0.0)
        new_attn_mask = ops.get_attention_mask(attn_mask, batch_size)
        out, last_attn_weights = ops.attention(query,
                                               key,
                                               value,
                                               casual_mask,
                                               -1e-4,
                                               attn_dropout,
                                               scale_attn_weights=True,
                                               training=training,
                                               attn_mask=new_attn_mask,
                                               head_mask=None)
        attn_weights_list.append(last_attn_weights)

        output = ops.merge_heads(out, num_heads, 1)

        return {"weighted_sum": output, "value": value}, attn_weights_list

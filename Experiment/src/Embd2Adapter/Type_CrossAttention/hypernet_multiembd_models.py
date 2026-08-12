import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple

from ..hypernet_utils import (
    AdapterLayersHyperNet,
    LayerNormHyperNet,
    SamplerOutput,
    TaskHyperNet,
)


AdapterOutput = namedtuple('AdapterOutput', ['up', 'down', 'pre_norm', 'post_norm'])

class AttentionPooling(nn.Module):
    def __init__(self, embd_dim: int):
        super().__init__()
        self.dim = embd_dim
        self.q_proj = nn.Linear(self.dim, self.dim)
        self.k_proj = nn.Linear(self.dim, self.dim)
        self.v_proj = nn.Linear(self.dim, self.dim)

    def forward(self, embeddings: torch.Tensor):
        batch_size, n, _ = embeddings.shape
        

class HyperNetController_MultiEmbds(nn.Module):
    def __init__(self, device: torch.device, embd_dim:int, projected_embd_dim:int, input_dim:int, reduction_factor:int, task_hidden_dim:int,num_layers:int = 28):
        super().__init__()
        self.num_layers = num_layers
        self.layer_norm_epsilon = 1e-6
        self.max_position_embeddings = 2
        self.device = device
        self.embd_dim = embd_dim

        self.embds_pool = AttentionPooling(self.embd_dim)

        self.layer_id_embeddings = nn.Embedding(self.num_layers,self.embd_dim).to(self.device)
        self.position_id_embeddings = nn.Embedding(2, self.embd_dim).to(self.device)

        self.vec_hypernet = TaskHyperNet(task_hidden_dim, projected_embd_dim, embd_dim*3)
        self.input_dim = input_dim
        self.down_sample_size = self.input_dim // reduction_factor

        self.up_sampler_hyper_net = AdapterLayersHyperNet(embd_dim=projected_embd_dim, input_dim=self.input_dim, output_dim=self.down_sample_size)
        self.down_sampler_hyper_net = AdapterLayersHyperNet(embd_dim=projected_embd_dim, input_dim=self.down_sample_size, output_dim=self.input_dim)

        self.pre_layernorm_hypernet = LayerNormHyperNet(projected_embd_dim, self.input_dim)
        self.post_layernorm_hypernet = LayerNormHyperNet(projected_embd_dim, self.input_dim)

    def pooling_emeddings(self, embeddings):
        self.embds_pool(embeddings)

    def concatinate_input(self, embedding, layer_id, position_id):
        layer_id_tensor = torch.tensor([layer_id], dtype=torch.long, device=self.device)
        layer_embedding = self.layer_id_embeddings(layer_id_tensor)
        position_id_tensor = torch.tensor([position_id], dtype=torch.long, device=self.device)
        position_embedding = self.position_id_embeddings(position_id_tensor)
        layer_embedding = layer_embedding.view(-1)
        position_embedding = position_embedding.view(-1)
        embeddings = torch.cat([embedding.view(1, -1), layer_embedding.view(1, -1), position_embedding.view(1, -1)], axis = 0)
        embeddings = self.vec_hypernet(embeddings.view(-1))
        return embeddings

    def forward(self, embeddings, layer_id):
        embedding = self.pooling_emeddings(embeddings=embeddings)
        feed_forward_embeddings = self.concatinate_input(embedding, layer_id, 0)
        self_attn_embeddings = self.concatinate_input(embedding, layer_id, 1)

        feed_forward_down = self.down_sampler_hyper_net(feed_forward_embeddings)
        feed_forward_up = self.up_sampler_hyper_net(feed_forward_embeddings)

        self_attn_down = self.down_sampler_hyper_net(self_attn_embeddings)
        self_attn_up = self.up_sampler_hyper_net(self_attn_embeddings)

        weight, bias = self.pre_layernorm_hypernet(feed_forward_embeddings)
        feed_forward_pre_norm = SamplerOutput(weight=weight, bias=bias)
        weight, bias = self.pre_layernorm_hypernet(self_attn_embeddings)
        self_attn_pre_norm = SamplerOutput(weight=weight, bias=bias)

        weight, bias = self.post_layernorm_hypernet(feed_forward_embeddings)
        feed_forward_post_norm = SamplerOutput(weight=weight, bias=bias)
        weight, bias = self.post_layernorm_hypernet(self_attn_embeddings)
        self_attn_post_norm = SamplerOutput(weight=weight, bias=bias)

        feed_forward_output = AdapterOutput(
            feed_forward_up, feed_forward_down,
            feed_forward_pre_norm, feed_forward_post_norm,
        )
        self_attn_output = AdapterOutput(
            self_attn_up, self_attn_down,
            self_attn_pre_norm, self_attn_post_norm,
        )

        return (feed_forward_output, self_attn_output)


class HyperNetWrapper_MultiEmbds(nn.Module):
    def __init__(self, original_layer: nn.Module, hypernet: HyperNetController_MultiEmbds, layer_id:int):
        super().__init__()
        self.original_layer = original_layer
        object.__setattr__(self, "_hypernet", hypernet)
        self.layer_id = layer_id

    def calc_adapter(self, hidden_states, parameters):
        residual = hidden_states
        hidden_states = F.layer_norm(
            hidden_states,
            (hidden_states.shape[-1],),
            parameters.pre_norm.weight,
            parameters.pre_norm.bias,
        )
        hidden_states = F.linear(
            hidden_states, parameters.up.weight.T, parameters.up.bias
        )
        hidden_states = F.relu(hidden_states)
        hidden_states = F.linear(
            hidden_states, parameters.down.weight.T, parameters.down.bias
        )
        hidden_states = F.layer_norm(
            hidden_states,
            (hidden_states.shape[-1],),
            parameters.post_norm.weight,
            parameters.post_norm.bias,
        )
        return residual + hidden_states

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=False,
        position_embeddings=None,
        embeddings=None,
        **kwargs,
    ):
        for embedding in embeddings:
            if embedding is None:
                raise ValueError("embedding must be passed to the model forward call.")
            if embedding.ndim == 2:
                if embedding.shape[0] != 1:
                    raise ValueError("HyperNet currently supports batch_size=1 only.")
                embedding = embedding[0]

        embeddings = embeddings.to(
            device=hidden_states.device,
            dtype=next(self._hypernet.parameters()).dtype,
        )
        feed_forward_parameters, self_attn_parameters = self._hypernet(
            embeddings, self.layer_id
        )

        residual = hidden_states
        hidden_states = self.original_layer.input_layernorm(hidden_states)
        hidden_states, _ = self.original_layer.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + self.calc_adapter(
            hidden_states, self_attn_parameters
        )

        residual = hidden_states
        hidden_states = self.original_layer.post_attention_layernorm(hidden_states)
        hidden_states = self.original_layer.mlp(hidden_states)
        hidden_states = residual + self.calc_adapter(
            hidden_states, feed_forward_parameters
        )
        return hidden_states

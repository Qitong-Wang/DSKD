 from typing import List, Optional, Union
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.processing_utils import Unpack
from transformers.utils import can_return_tuple, auto_docstring
from transformers.models.llama.modeling_llama import LlamaModel, LlamaPreTrainedModel, KwargsForCausalLM
from transformers import GenerationMixin
import torch
from transformers.cache_utils import Cache
from transformers.utils import logging
logger = logging.get_logger(__name__)
import torch.nn.functional as F
import os
from torch import nn
 
from transformers.models.mistral.modeling_mistral import MistralModel, MistralPreTrainedModel, KwargsForCausalLM


def replace_sense(input_emb, sense_tensor_emb, type):
    # Ensure input_emb is at least 2D: (1, dim) if originally (dim,)
    if input_emb.dim() == 1:
        input_emb = input_emb.unsqueeze(0)  # (1, dim)
    if torch.is_tensor(sense_tensor_emb) and sense_tensor_emb.dim() == 1:
        return None, sense_tensor_emb
    if type == "dot":
        result = torch.matmul(input_emb, sense_tensor_emb.T)  # (batch, num_sense)
        index = torch.argmax(result, dim=1)  # (batch,)
    elif type == "l2":
        diff = input_emb.unsqueeze(1) - sense_tensor_emb.unsqueeze(0)
        result = torch.norm(diff, dim=2)  # (batch, num_sense)
        index = torch.argmin(result, dim=1)  # (batch,)
    elif type == "cos":
        tensor1_normalized = F.normalize(input_emb, p=2, dim=1)  # (batch, dim)
        tensor2_normalized = F.normalize(sense_tensor_emb, p=2, dim=1)  # (num_sense, dim)
        result = torch.matmul(tensor1_normalized, tensor2_normalized.T)  # (batch, num_sense)
        index = torch.argmax(result, dim=1)  # (batch,)
    else:
        raise ValueError("Unsupported type")
    output = sense_tensor_emb[index]  # (batch, dim)
    if output.shape[0] == 1:
        return index.squeeze(0), output.squeeze(0)
    return index, output


@auto_docstring
class SKDLlamaForCausalLM(LlamaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}
    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.run_option = None
        self.sense_tensor = None
        self.post_init()
    def set_training_args(self, training_args):
        self.run_option = training_args['run_option']
        self.loss_type = training_args['loss_type']
    def get_input_embeddings(self):
        return self.model.embed_tokens
    def set_input_embeddings(self, value):
        self.model.embed_tokens = value
    def get_output_embeddings(self):
        return self.lm_head
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
    def set_decoder(self, decoder):
        self.model = decoder
    def get_decoder(self):
        return self.model
    def set_sense_tensor(self, sense_tensor, device):
        if device == "cpu":
            self.sense_tensor = sense_tensor
        else:
            self.sense_tensor = sense_tensor.to(device)
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> CausalLMOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
        if self.run_option == "gather":
            self.extra_info.setdefault('hidden_state', [])
            self.extra_info['hidden_state'].append(hidden_states[0, :, :].detach().to('cpu').to(torch.float32).numpy())
            self.extra_info.setdefault('token_id', [])
            self.extra_info['token_id'].append(torch.argmax(self.lm_head(hidden_states[0, :, :]), dim=-1))

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


    def generate_with_info(self, *args, **kwargs):
        self.extra_info = {}
        sequences = self.generate(*args, **kwargs)
        return sequences, self.extra_info




@auto_docstring
class SKDMistralForCausalLM(MistralPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = MistralModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.run_option = None
        self.sense_tensor = None
        self.post_init()

    def set_training_args(self, training_args):
        self.run_option = training_args['run_option']
        self.loss_type = training_args['loss_type']

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model


    def set_sense_tensor(self, sense_tensor, device):
        if device == "cpu":
            self.sense_tensor = sense_tensor
        else:
            self.sense_tensor = sense_tensor.to(device)

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> CausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, MistralForCausalLM

        >>> model = MistralForCausalLM.from_pretrained("meta-mistral/Mistral-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-mistral/Mistral-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state


        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        if self.run_option == "gather":
            self.extra_info.setdefault('hidden_state', [])
            self.extra_info['hidden_state'].append(hidden_states[0, :, :].detach().to('cpu').to(torch.float32).numpy())
            self.extra_info.setdefault('token_id', [])
            self.extra_info['token_id'].append(torch.argmax(self.lm_head(hidden_states[0, :, :]), dim=-1))

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    def generate_with_info(self, *args, **kwargs):
        self.extra_info = {}
        sequences = self.generate(*args, **kwargs)
        return sequences, self.extra_info
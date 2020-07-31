"""
Run the model with integrated and intermediate gradients
"""

from captum.attr import LayerIntegratedGradients
import logging
import torch

from intermediate_gradients.layer_intermediate_gradients import LayerIntermediateGradients


def prepare_input(sentence, tokenizer):
    """
    Tokenize, truncate, and prepare the input for modeling.
    NOTE: Requires Transformers>=3.0.0

    Parameters
    ----------
    sentence: str
        Input sentence to obtain sentiment from.
    tokenizer: XLNetTokenizer
        Tokenizer for tokenizing input.

    Returns
    -------
    features: dict
        Keys
        ----
        input_ids: torch.tensor(1, num_ids), dtype=torch.int64
            Tokenized sequence text.
        token_type_ids: torch.tensor(1, num_ids), dtype=torch.int64
            Token type ids for the inputs.
        attention_mask: torch.tensor(1, num_ids), dtype=torch.int64
            Masking tensor for the inputs.
    """
    features = tokenizer([sentence], return_tensors='pt', truncation=True, max_length=512)
    return features


def bert_sequence_forward_func(inputs, model, tok_type_ids, att_mask):
    """
    Passes forward the inputs and relevant keyword arguments.
    Parameters
    ----------
    inputs: torch.tensor(1, num_ids), dtype=torch.int64
        Encoded form of the input sentence.
    tok_type_ids: torch.tensor(1, num_ids), dtype=torch.int64
        Tensor to specify token type for the model.
        Because sentiment analysis uses only one input, this is just a tensor of zeros.
    att_mask: torch.tensor(1, num_ids), dtype=torch.int64
        Tensor to specify attention masking for the model.
    Returns
    -------
    outputs: torch.tensor(1, 2), dtype=torch.float32
        Output classifications for the model.
    """
    outputs = model(inputs, token_type_ids=tok_type_ids, attention_mask=att_mask)
    return outputs


def xlnet_sequence_forward_func(inputs, model, tok_type_ids, att_mask):
    """
    Passes forward the inputs and relevant keyword arguments.
    Parameters
    ----------
    inputs: torch.tensor(1, num_ids), dtype=torch.int64
        Encoded form of the input sentence.
    tok_type_ids: torch.tensor(1, num_ids), dtype=torch.int64
        Tensor to specify token type for the model.
        Because sentiment analysis uses only one input, this is just a tensor of zeros.
    att_mask: torch.tensor(1, num_ids), dtype=torch.int64
        Tensor to specify attention masking for the model.
    Returns
    -------
    outputs: torch.tensor(1, 2), dtype=torch.float32
        Output classifications for the model.
    """
    outputs = model(inputs, token_type_ids=tok_type_ids, attention_mask=att_mask)[0]
    return outputs


def run_models(model_name, model, tokenizer, sequence, device):
    """
    Run Integrated and Intermediate gradients on the model layer.

    Parameters
    ----------
    model_name: str
       Name of the model that is being run.
       Currently supported are "Bert" or "XLNet"
    model: torch.nn.Module
       Module to run 
    tokenizer: transformers.tokenizer
       Tokenizer to tokenize the sequence
    sequence: str
       Sequence to get the gradients from.
    Device: torch.device
       Device that models are stored on.

    Returns
    -------
    gradients_dict: dict
        Dictionary containing the gradient tensors with the following keys:
        "integrated_gradients", "intermediate_gradients", "step_sizes", and "intermediates".
    """
    if model_name == "bert":
        layer_interm = LayerIntermediateGradients(bert_sequence_forward_func, model.bert.embeddings)
        lig = LayerIntegratedGradients(bert_sequence_forward_func, model.bert.embeddings)
    elif model_name == "xlnet":
        layer_interm = LayerIntermediateGradients(
            xlnet_sequence_forward_func, model.transformer.batch_first
        )
        lig = LayerIntegratedGradients(xlnet_sequence_forward_func, model.transformer.batch_first)

    features = prepare_input(sequence, tokenizer)
    input_ids = features["input_ids"].to(device)
    token_type_ids = features["token_type_ids"].to(device)
    attention_mask = features["attention_mask"].to(device)

    grads, step_sizes, intermediates = layer_interm.attribute(inputs=input_ids,
                                                              baselines=baseline_ids,
                                                              additional_forward_args=(
                                                                  model,
                                                                  token_type_ids,
                                                                  attention_mask
                                                              ),
                                                              target=1,
                                                              n_steps=n_steps)

    integrated_grads = lig.attribute(inputs=input_ids,
                                     baselines=baseline_ids,
                                     additional_forward_args=(
                                         model,
                                         token_type_ids,
                                         attention_mask
                                     ),
                                     target=1,
                                     n_steps=n_steps)

    grads_dict = {"intermediate_grads": grads.to("cpu"),
                  "step_sizes": step_sizes.to("cpu"),
                  "intermediates": intermediates.to("cpu")
                  "integrated_grads": integrated_grads.to("cpu")}

    return grads_dict
                                                     

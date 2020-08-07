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


def generate_xlnet_baselines(baseline, input_ids, tokenizer):
    """
    Produce the desired baseline id tensor for integrated gradients.

    Parameters
    ----------
    baseline: str
        Baseline to run with integrated gradients. Currently supported are 'zero', 'pad', 'unk',
       'rand-norm', 'rand-unif', and 'period'.
    input_ids: torch.tensor(1, num_ids) dtype=torch.int64
        Encoded form of the input sentence.
    tokenizer: transformers.tokenizer
       Tokenizer to process the sequence and produce the input ids

    Returns
    -------
    baseline_ids: torch.tensor(1, num_ids) dtype=torch.int64
        Tensor of the baseline token ids in the same shape as input_ids.
    """
    if baseline == "pad":
        baseline_token_id = tokenizer.pad_token_id
    elif baseline == "unk":
        baseline_token_id = tokenizer.unk_token_id
    elif baseline == "period":
        baseline_token_id = tokenizer.encoder('.', add_special_tokens=False)[0]
    elif baseline == "zero":
        baseline_token_id = 32000
    elif baseline == "rand-unif":
        baseline_token_id = 32001
    elif baseline == "rand-norm":
        baseline_token_id = 32002
    baseline_ids = torch.ones(input_ids.shape, dtype=torch.int64) * baseline_token_id
    return baseline_ids


def generate_bert_baselines(baseline, input_ids, tokenizer):
    """
    Produce the desired baseline id tensor for integrated gradients.

    Parameters
    ----------
    baseline: str
        Baseline to run with integrated gradients. Currently supported are 'zero', 'pad', 'unk',
       'rand-norm', 'rand-unif', and 'period'.
    input_ids: torch.tensor(1, num_ids) dtype=torch.int64
        Encoded form of the input sentence.
    tokenizer: transformers.tokenizer
       Tokenizer to process the sequence and produce the input ids.

    Returns
    -------
    baseline_ids: torch.tensor(1, num_ids) dtype=torch.int64
        Tensor of the baseline token ids in the same shape as input_ids.
    """
    if baseline == "pad":
        baseline_token_id = tokenizer.pad_token_id
    elif baseline == "unk":
        baseline_token_id = tokenizer.unk_token_id
    elif baseline == "period":
        baseline_token_id = tokenizer.encoder('.', add_special_tokens=False)[0]
    elif baseline == "zero":
        baseline_token_id = 30522
    elif baseline == "rand-unif":
        baseline_token_id = 30524
    elif baseline == "rand-norm":
        baseline_token_id = 30523
    baseline_ids = torch.ones(input_ids.shape, dtype=torch.int64) * baseline_token_id
    return baseline_ids


def run_models(model_name, model, tokenizer, sequence, device, baseline):
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
       Tokenizer to process the sequence and produce the input ids
    sequence: str
       Sequence to get the gradients from.
    device: torch.device
       Device that models are stored on.
    baseline: str
       Baseline to run with integrated gradients. Currently supported are 'zero', 'pad', 'unk',
       'rand-norm', 'rand-unif', and 'period'.

    Returns
    -------
    gradients_dict: dict
        Dictionary containing the gradient tensors with the following keys:
        "integrated_gradients", "intermediate_gradients", "step_sizes", and "intermediates".
    """
    features = prepare_input(sequence, tokenizer)
    input_ids = features["input_ids"].to(device)
    token_type_ids = features["token_type_ids"].to(device)
    attention_mask = features["attention_mask"].to(device)

    # set up gradients and the baseline ids
    if model_name == "bert":
        layer_interm = LayerIntermediateGradients(bert_sequence_forward_func, model.bert.embeddings)
        lig = LayerIntegratedGradients(bert_sequence_forward_func, model.bert.embeddings)
        baseline_ids = generate_bert_baselines(baseline, input_ids, tokenizer).to(device)
    elif model_name == "xlnet":
        layer_interm = LayerIntermediateGradients(
            xlnet_sequence_forward_func, model.transformer.batch_first
        )
        lig = LayerIntegratedGradients(xlnet_sequence_forward_func, model.transformer.batch_first)
        baseline_ids = generate_xlnet_baselines(baseline, input_ids, tokenizer).to(device)

    grads, step_sizes, intermediates = layer_interm.attribute(inputs=input_ids,
                                                              baselines=baseline_ids,
                                                              additional_forward_args=(
                                                                  model,
                                                                  token_type_ids,
                                                                  attention_mask
                                                              ),
                                                              target=1,
                                                              n_steps=50) # maybe pass n_steps as CLI argument

    integrated_grads = lig.attribute(inputs=input_ids,
                                     baselines=baseline_ids,
                                     additional_forward_args=(
                                         model,
                                         token_type_ids,
                                         attention_mask
                                     ),
                                     target=1,
                                     n_steps=50)

    grads_dict = {"intermediate_grads": grads.to("cpu"),
                  "step_sizes": step_sizes.to("cpu"),
                  "intermediates": intermediates.to("cpu"),
                  "integrated_grads": integrated_grads.to("cpu")}

    return grads_dict
                                                     

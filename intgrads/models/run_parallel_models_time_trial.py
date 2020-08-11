"""
CLI to convert line-separated .txt file of sequences to a bash file of curl commands for int-gradients-server
"""

import click
import json
import os
import time
import torch

from torch import nn

from transformers import BertTokenizer, XLNetTokenizer

from captum.attr import LayerIntegratedGradients
from intermediate_gradients.layer_intermediate_gradients import LayerIntermediateGradients

from modified_xlnet import XLNetForSequenceClassification
from bert_model import BertForSequenceClassification, BertConfig
from modified_xlnet_parallel import XLNetForSequenceClassification as XLNetForSequenceClassificationParallel
from bert_model_parallel import BertForSequenceClassification as BertForSequenceClassificationParallel

from tqdm import tqdm

def load_bert_model(model_path, device, num_cuda_devs):
    """
    Load the pretrained BERT model states and prepare the model for sentiment analysis on CPU.
    
    This method returns a custom BertForSequenceClassification model that allows it to work
    with LayerIntegratedGradients and LayerIntermediateGradients.
    Loads a slighly different parallel model if num_cuda_devs > 1.

    Parameters
    ----------
    model_path: str
        Path to the pretrained model states binary file.
    device: torch.device
        Device to load the model on.
    num_cuda_devs: int
        Determines how model parallel is used.

    Returns
    -------
    model: BertForSequenceClassification
        Model with the loaded pretrained states.
    tokenizer: BertTokenizer
        Instance of the tokenizer for BERT models.
    """
    config = BertConfig(vocab_size=30522, type_vocab_size=2)
    if num_cuda_devs < 2:
        model = BertForSequenceClassification(config, 2, [11])
        model_states = torch.load(model_path, map_location=torch.device("cpu"))
        model.load_state_dict(model_states)

        model.eval()
        model.to(device) # puts it on cuda:0 or cpu, less work

    else:
        if num_cuda_devs == 2:
            embed_device = "cuda:0"
            encoder_device1 = "cuda:1"
            encoder_device2 = "cuda:1"
            encoder_device3 = "cuda:0"
            pooler_device = "cuda:0"
        elif num_cuda_devs == 3:
            embed_device = "cuda:0"
            encoder_device1 = "cuda:0"
            encoder_device2 = "cuda:1"
            encoder_device3 = "cuda:2"
            pooler_device = "cuda:0"
        else:
            # 4 cuda devices
            embed_device = "cuda:0"
            encoder_device1 = "cuda:1"
            encoder_device2 = "cuda:2"
            encoder_device3 = "cuda:3"
            pooler_device = "cuda:0"

        model = BertForSequenceClassificationParallel(config, 2, [11],
                                                      embeddings_device=embed_device,
                                                      encoder_device1=encoder_device1,
                                                      encoder_device2=encoder_device2,
                                                      encoder_device3=encoder_device3,
                                                      pooler_device=pooler_device)
        model_states = torch.load(model_path, map_location="cpu")
        model.load_state_dict(model_states, strict=False) 
        model.eval()

    # override the embeddings layer
    weight = torch.zeros((30525, 768), dtype=torch.float32)
    weight[0:30522, :] = model.bert.embeddings.word_embeddings.weight
    weight[30523, :] = torch.rand(768, dtype=torch.float32)
    weight[30524, :] = torch.randn(768, dtype=torch.float32)
    weight = weight.to(model.bert.embeddings.word_embeddings.weight.device)
    model.bert.embeddings.word_embeddings.weight = nn.Parameter(weight)

    # override the config number of embeddings
    model.bert.embeddings.word_embeddings.num_embeddings += 3

    tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
    return model, tokenizer


def load_xlnet_base_model(model_path, device, num_cuda_devs):
    """
    Load the pretrained xlnet states and prepare the model for sentiment analysis.

    Loads a slighly different parallel model if num_cuda_devs > 1.

    Parameters
    ----------
    model_path: str
        Path to the pretrained model states binary file.
    device: torch.device
        Device to load the model on.
    num_cuda_devs: int
        Determines how model parallel is used.

    Returns
    -------
    model: XLNetForSequenceClassification
        Model with the loaded pretrained states.
    """
    if num_cuda_devs < 2:
        model = XLNetForSequenceClassification.from_pretrained("xlnet-base-cased")
        model_states = torch.load(model_path, map_location=device)
        model.load_state_dict(model_states)

        model.eval()
        model.to(device)

    else:
        if num_cuda_devs == 2:
            embed_device = "cuda:0"
            encoder_device1 = "cuda:0"
            encoder_device2 = "cuda:1"
            encoder_device3 = "cuda:1"
            logits_device = "cuda:0"
        elif num_cuda_devs == 3:
            embed_device = "cuda:0"
            encoder_device1 = "cuda:0"
            encoder_device2 = "cuda:1"
            encoder_device3 = "cuda:2"
            logits_device = "cuda:0"
        else:
            # 4 cuda devices
            embed_device = "cuda:0"
            encoder_device1 = "cuda:1"
            encoder_device2 = "cuda:2"
            encoder_device3 = "cuda:3"
            logits_device = "cuda:0"

        model = XLNetForSequenceClassificationParallel.from_pretrained("xlnet-base-cased",
                                                                       embeddings_device=embed_device,
                                                                       encoder_device1=encoder_device1,
                                                                       encoder_device2=encoder_device2,
                                                                       encoder_device3=encoder_device3,
                                                                       logits_device=logits_device)
        model_states = torch.load(model_path, map_location="cpu")
        model.load_state_dict(model_states)
        model.eval()

    # override the embeddings layer
    weight = torch.zeros((32003, 768), dtype=torch.float32)
    weight[0:32000, :] = model.transformer.word_embedding.weight
    weight[32001, :] = torch.rand(768, dtype=torch.float32)
    weight[32002, :] = torch.randn(768, dtype=torch.float32)
    weight = weight.to(model.transformer.word_embedding.weight.device)
    model.transformer.word_embedding.weight = nn.Parameter(weight)

    # override the config number of embeddings
    model.transformer.word_embedding.num_embeddings += 3

    tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")
    return model, tokenizer


def load_and_tokenize(filepath, tokenizer):
    """
    Loads the .txt file and tokenizes each output, storing them in a list
    """
    tokenized_examples = []
    with open(filepath, 'r') as fobj:
        lines = fobj.readlines()
    for line in tqdm(lines):
        features = tokenizer([line], return_tensors='pt', truncation=True, max_length=512)
        features["baseline_ids"] = torch.zeros(features["input_ids"].shape, dtype=torch.int64)
        tokenized_examples.append(features)
    return tokenized_examples


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


def run_models(model, model_name, num_trials, subset, tokenized_list, device):
    if model_name == "bert":
        layer_interm = LayerIntermediateGradients(bert_sequence_forward_func, model.bert.embeddings)
        lig = LayerIntegratedGradients(bert_sequence_forward_func, model.bert.embeddings)
    elif model_name == "xlnet":
        layer_interm = LayerIntermediateGradients(
            xlnet_sequence_forward_func, model.transformer.batch_first
        )
        lig = LayerIntegratedGradients(xlnet_sequence_forward_func, model.transformer.batch_first)

    run_through_example = tokenized_list[-1]
    tokenized_list = tokenized_list[:subset]

    input_ids = run_through_example["input_ids"].to(device)
    token_type_ids = run_through_example["token_type_ids"].to(device)
    attention_mask = run_through_example["attention_mask"].to(device)
    baseline_ids = run_through_example["baseline_ids"].to(device)

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

    for repetition in tqdm(range(num_trials)):
        start_time = time.perf_counter()
        for feature in tokenized_list:
            input_ids = feature["input_ids"].to(device)
            token_type_ids = feature["token_type_ids"].to(device)
            attention_mask = feature["attention_mask"].to(device)
            baseline_ids = feature["baseline_ids"].to(device)

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
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print("Repetition %s Elapsed Time for %s examples: "%(repetition, subset), elapsed_time)


@click.command()
@click.option(
    "-fp",
    "--filepath",
    help="path to the .txt file of sequences",
    required=True
)
@click.option(
    "-s",
    "--subset",
    help="Truncate the list of sequences to this value. Default is -1.",
    required=False,
    default=-1,
)
@click.option(
    "-xlp",
    "--xlnet-model-path",
    help="Path to the XLNet Base model.",
    required=False,
    default=None,
)
@click.option(
    "-bp",
    "--bert-model-path",
    help="Path to the BERT model.",
    required=False,
    default=None,
)
@click.option(
    "-cudanum",
    "--num-cuda-devices",
    help="Number of cuda devices to run models on",
    default=0
)
@click.option(
    "--num-trials",
    "-nt",
    help="Number of times to repeat the process",
    default=3,
)
def main(filepath, subset, xlnet_model_path=None, bert_model_path=None, num_cuda_devices=0, num_trials=3):

    if int(num_cuda_devices) != 0:
        DEVICE = torch.device("cuda:0")
        # will always load inputs on this device even if model parallel option used
    else:
        DEVICE = torch.device("cpu")
    if xlnet_model_path is not None:
        model, tokenizer = load_xlnet_base_model(str(xlnet_model_path), DEVICE, int(num_cuda_devices))
        model_name = "xlnet"
    elif bert_model_path is not None:
        model, tokenizer = load_bert_model(str(bert_model_path), DEVICE, int(num_cuda_devices))
        model_name = "bert"

    tokenized_examples = load_and_tokenize(str(filepath), tokenizer)

    run_models(model, model_name, int(num_trials), int(subset), tokenized_examples, DEVICE)

if __name__ == "__main__":
    main()

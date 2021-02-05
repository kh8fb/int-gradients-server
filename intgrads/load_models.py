"""
Interface for loading the supported models and tokenizers.
"""

from collections import OrderedDict
import logging
from transformers import BertTokenizer, XLNetTokenizer
import torch
from torch import nn

from .models.modified_xlnet import XLNetForSequenceClassification
from .models.bert_model import BertForSequenceClassification, BertConfig
from .models.modified_xlnet_parallel import XLNetForSequenceClassification as XLNetForSequenceClassificationParallel
from .models.bert_model_parallel import BertForSequenceClassification as BertForSequenceClassificationParallel


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
        if list(model_states.keys())[0][:6] == "module":
            for key in list(model_states.keys()):
                new_name = key[7:]
                model_states[new_name] = model_states[key]
                del model_states[key]
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


def load_xlnet_large_model(model_path, device, num_cuda_devs):
    """
    Load the pretrained xlnet-large states and prepare the model for sentiment analysis.

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
        model = XLNetForSequenceClassification.from_pretrained("xlnet-large-cased")
        model_states = torch.load(model_path, map_location=device)
        if list(model_states.keys())[0][:6] == "module":
            for key in list(model_states.keys()):
                new_name = key[7:]
                model_states[new_name] = model_states[key]
                del model_states[key]
        model.load_state_dict(model_states)

        model.eval()
        model.to(device)

    else:
        if num_cuda_devs == 2:
            embed_device="cuda:0"
            encoder_device1="cuda:0"
            encoder_device2="cuda:1"
            encoder_device3="cuda:1"
            logits_device="cuda:0"
        elif num_cuda_devs ==3:
            embed_device="cuda:0"
            encoder_device1="cuda:0"
            encoder_device2="cuda:1"
            encoder_device3="cuda:2"
            logits_device="cuda:0"
        else:
            # 4 cuda devices
            embed_device="cuda:0"
            encoder_device1="cuda:1"
            encoder_device2="cuda:2"
            encoder_device3="cuda:3"
            logits_device="cuda:0"

        model = XLNetForSequenceClassificationParallel("xlnet-large-cased",
                                                       embeddings_device=embed_device,
                                                       encoder_device1=encoder_device1,
                                                       encoder_device2=encoder_device2,
                                                       encoder_device3=encoder_device3,
                                                       logits_device=logits_device)
        model_states = torch.load(model_path, map_location="cpu")
        model.load_state_dict(model_states)
        model.eval()

    tokenizer = XLNetTokenizer.from_pretrained("xlnet-large-cased")
    return model, tokenizer


def load_models(device, num_cuda_devs, bert_path, xlnet_base_path, xlnet_large_path):
    """
    Load the models and tokenizers and return them in a dictionary.

    Parameters
    ----------
    cuda: bool
        Whether or not to run models on CUDA.
    num_cuda_devs: int
        Determines how model parallel is used.
    bert_path: str or None
        Path to the pretrained BERT model states binary file.
    xlnet_base_path: str or None
        Path to the pretrained XLNet base model states binary file.
    xlnet_large_path: str or None
        Path to the pretrained XLNet large model states binary file.

    Returns
    -------
    model_dict: dict
        Dictionary with storing each of the model's ids and tokenizers.
        Current keys are 'xlnet' and 'bert'.
    """
    logging.basicConfig(level=logging.ERROR) # disable model warning messages

    if bert_path is not None:
        bert_model, bert_tokenizer = load_bert_model(str(bert_path), device, num_cuda_devs)
        return {"model_name": "bert", "model": bert_model, "tokenizer": bert_tokenizer}
    elif xlnet_base_path is not None:
        xlnet_model, xlnet_tokenizer = load_xlnet_base_model(str(xlnet_base_path), device, num_cuda_devs)
        return {"model_name": "xlnet", "model": xlnet_model, "tokenizer": xlnet_tokenizer}
    elif xlnet_large_path is not None:
        xlnet_model, xlnet_tokenizer = load_xlnet_large_model(str(xlnet_large_path), device, num_cuda_devs)
        return {"model_name": "xlnet", "model": xlnet_model, "tokenizer": xlnet_tokenizer}

    ## Add additional models here
    else:
        return None

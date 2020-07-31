"""
Interface for loading the supported models and tokenizers.
"""

from collections import OrderedDict
import logging
from transformers import BertTokenizer, XLNetTokenizer
import torch

from .modified_xlnet import XLNetForSequenceClassification
from .bert_model import BertForSequenceClassification, BertConfig


def load_bert_model(model_path, device):
    """
    Load the pretrained BERT model states and prepare the model for sentiment analysis on CPU.
    
    This method returns a custom BertForSequenceClassification model that allows it to work
    with LayerIntegratedGradients and LayerIntermediateGradients.

    Parameters
    ----------
    model_path: str
        Path to the pretrained model states binary file.
    device: torch.device
        Device to load the model on.

    Returns
    -------
    model: BertForSequenceClassification
        Model with the loaded pretrained states.
    tokenizer: BertTokenizer
        Instance of the tokenizer for BERT models.
    """
    config = BertConfig(vocab_size=30522, type_vocab_size=2)
    model = BertForSequenceClassification(config, 2, [11])
    model_states = torch.load(model_path, map_location=torch.device("cpu"))
    model.load_state_dict(model_states)

    model.eval()
    model.to(device)

    tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
    return model, tokenizer


def load_xlnet_model(model_path, device):
    """
    Load the pretrained xlnet states and prepare the model for sentiment analysis.

    Parameters
    ----------
    model_path: str
        Path to the pretrained model states binary file.
    device: torch.device
        Device to load the model on.

    Returns
    -------
    model: XLNetForSequenceClassification
        Model with the loaded pretrained states.
    """
    model = XLNetForSequenceClassification.from_pretrained("xlnet-base-cased")
    model_states = torch.load(model_path, map_location=device)
    new_model_states = OrderedDict()
    for state in model_states:
        correct_state = state[7:]
        new_model_states[correct_state] = model_states[state]
    model.load_state_dict(new_model_states)

    model.eval()
    model.to(device)

    tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")
    return model, tokenizer


def load_models(device, bert_path, xlnet_path):
    """
    Load the models and tokenizers and return them in a dictionary.

    Parameters
    ----------
    cuda: bool
        Whether or not to run models on CUDA.
    bert_path: str or None
        Path to the pretrained BERT model states binary file.
    xlnet_path: str or None
        Path to the pretrained XLNet model states binary file.

    Returns
    -------
    model_dict: dict
        Dictionary with storing each of the model's ids and tokenizers.
        Current keys are 'xlnet' and 'bert'.
    """
    logging.basicConfig(level=logging.ERROR) # disable model warning messages

    if bert_path is not None:
        bert_model, bert_tokenizer = load_bert_model(str(bert_path), device)
    else:
        bert_model, bert_tokenizer = None, None

    if xlnet_path is not None:
        xlnet_model, xlnet_tokenizer = load_xlnet_model(str(xlnet_path), device)
    else:
        xlnet_model, xlnet_tokenizer = None, None

    ## Add additional models here

    models_dict = {"xlnet": (xlnet_model, xlnet_tokenizer),
                   "bert": (bert_model, bert_tokenizer)}
    return models_dict

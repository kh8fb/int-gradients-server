"""
Server for obtaining integrated and intermediate gradients on specific models.
"""

import click
from collections import OrderedDict
from flask import Flask, request, send_file
import gzip
from io import BytesIO
import shutil
import torch


from . import cli_main
from .load_models import load_models
from .run_models import run_models

app = Flask(__name__)
MODEL_DICT = {}
DEVICE = None


@app.route("/xlnet/", methods=['POST'])
def run_xlnet():
    """
    Obtain the gradients from running the finetuned XLNet model on the sequence.
    The outputs are saved as a gzipped dictionary with the keys:
    integrated_gradients, intermediate_gradients, step_sizes, sentiment

    The sequence to run gradients on should be passed in JSON format through the POST request.
    """
    if request.method == 'POST':

        data = request.get_json(force=True)

        sequence = data["sequence"]

        grads_dict = run_models("xlnet",
                                MODEL_DICT["xlnet"][0],
                                MODEL_DICT["xlnet"][1],
                                sequence,
                                DEVICE)

        temp_bytes, temp_gzip = BytesIO(), BytesIO()

        torch.save(grads_dict, temp_bytes)
        temp_bytes.seek(0)

        with gzip.GzipFile(fileobj=temp_gzip, mode='wb') as f_out:
            shutil.copyfileobj(temp_bytes, f_out)
            # I assume this is the fastest option
        temp_gzip.seek(0)

        return send_file(temp_gzip, as_attachment=True, mimetype="/application/gzip", attachment_filename="returned_gradients.gzip")


@app.route("/bert/<sequence>", methods=['POST'])
def run_bert(sequence):
    """
    Obtain the gradients from running the finetuned BERT model on the sequence.
    The outputs are saved as a gzipped dictionary with the keys:
    integrated_gradients, intermediate_gradients, step_sizes, sentiment

    The sequence to run gradients on should be passed in JSON format through the POST request.
    """
    if request.method == 'POST':

        data = request.get_json(force=True)

        sequence = data["sequence"]

        grads_dict = run_models("bert",
                                MODEL_DICT["bert"][0],
                                MODEL_DICT["bert"][1],
                                sequence,
                                DEVICE)

        temp_bytes, temp_gzip = BytesIO(), BytesIO()

        torch.save(grads_dict, temp_bytes)
        temp_bytes.seek(0)

        with gzip.GzipFile(fileobj=temp_gzip, mode='wb') as f_out:
            shutil.copyfileobj(temp_bytes, f_out)
            # I assume this is the fastest option
        temp_gzip.seek(0)

        return send_file(temp_gzip, as_attachment=True, mimetype="/application/gzip", attachment_filename="returned_gradients.gzip")


@cli_main.command(help="Start a server and initialize the models for calculating gradients.")
@click.option(
    "-h",
    "--host",
    default="localhost",
    help="Host to bind to."
)
@click.option(
    "-p",
    "--port",
    default=8888,
    help="Port to bind to."
)
@click.option(
    "--cuda/--cpu",
    default=True,
    help="Whether or not to run models on CUDA."
)
@click.option(
    "--bert-path",
    "-bp",
    help="Path to the BERT finetuned model.",
    default=None,
)
@click.option(
    "--xlnet-path",
    "-xl",
    help="Path to the XLNet model.",
    default=None
)
def serve(host, port, cuda, bert_path=None, xlnet_path=None):
    global MODEL_DICT, DEVICE
    MODEL_DICT = load_models(cuda, bert_path, xlnet_path)

    if cuda:
        DEVICE = torch.device("cuda:0")
    else:
        DEVICE = torch.device("cpu")

    app.run(host=host, port=port, debug=True)

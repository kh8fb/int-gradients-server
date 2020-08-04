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

@app.route("/model/", methods=["POST"])
def run_model():
    """
    Obtain the gradients from running the specified model on the sequence.
    The outputs are saved as a gzipped dictionary with the keys:
    integrated_gradients, intermediate_gradients, step_sizes, intermediates.
    """
    if request.method == 'POST':
        data = request.get_json(force=True)

        sequence = data["sequence"]

        grads_dict = run_models(MODEL_DICT["model_name"],
                                MODEL_DICT["model"],
                                MODEL_DICT["tokenizer"],
                                sequence,
                                DEVICE)

        temp_bytes, temp_gzip = BytesIO(), BytesIO()

        torch.save(grads_dict, temp_bytes)
        temp_bytes.seek(0)

        with gzip.GzipFile(fileobj=temp_gzip, mode='wb') as f_out:
            shutil.copyfileobj(temp_bytes, f_out)

        temp_gzip.seek(0)

        return send_file(temp_gzip, as_attachment=True, mimetype="/application/gzip", attachment_filename="returned_gradients.gzip")


@cli_main.command(help="Start a server and initialize the models for calculating gradients.")
@click.option(
    "-h",
    "--host",
    default="localhost",
    help="Host to bind to. Default localhost"
)
@click.option(
    "-p",
    "--port",
    default=8888,
    help="Port to bind to. Default 8888"
)
@click.option(
    "--cuda/--cpu",
    default=True,
    help="Whether or not to run models on CUDA."
)
@click.option(
    "--bert-path",
    "-bp",
    help="Path to the BERT finetuned model. Specify only one model path.",
    default=None,
)
@click.option(
    "--xlnet-base-path",
    "-xlb",
    help="Path to the XLNet base model. Specifiy only one model path.",
    default=None,
)
@click.option(
    "--xlnet-large-path",
    "-xll",
    help="Path to the XLNet large model. Specifiy only one model path.",
    default=None,
)
def serve(host, port, cuda, bert_path=None, xlnet_base_path=None, xlnet_large_path=None):
    global MODEL_DICT, DEVICE

    if cuda:
        DEVICE = torch.device("cuda:0")
    else:
        DEVICE = torch.device("cpu")

    MODEL_DICT = load_models(DEVICE, bert_path, xlnet_base_path, xlnet_large_path)

    app.run(host=host, port=port, debug=True)

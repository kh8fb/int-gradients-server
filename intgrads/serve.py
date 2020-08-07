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
BASELINE = ""
DEVICE = None

@app.route("/model/", methods=["POST"])
def run_model():
    """
    Obtain the gradients from running the specified model on the sequence.
    The outputs are saved as a gzipped dictionary with the keys:
    integrated_gradients, intermediate_gradients, step_sizes, intermediates.
    """
    if request.method == 'POST':
        #data = request.get_json(force=True)
        print(request)
        data = request.json

        sequence = data["sequence"]

        grads_dict = run_models(MODEL_DICT["model_name"],
                                MODEL_DICT["model"],
                                MODEL_DICT["tokenizer"],
                                sequence,
                                DEVICE,
                                BASELINE)

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
    required=False,
    default="localhost",
    help="Host to bind to. Default localhost"
)
@click.option(
    "-p",
    "--port",
    default=8888,
    required=False,
    help="Port to bind to. Default 8888"
)
@click.option(
    "--cuda/--cpu",
    required=True,
    default=True,
    help="Whether or not to run models on CUDA."
)
@click.option(
    "-b",
    "--baseline",
    required=True,
    help="""Baseline to run with integrated gradients.
    Currently supported are 'zero', 'pad', 'unk', 'rand-norm', 'rand-unif', and 'period'.""",
)
@click.option(
    "--num-cuda-devs",
    default=1,
    required=False,
    help="Number of cuda devices to run the model on. Should be between 1 and 4.",
)
@click.option(
    "--bert-path",
    "-bp",
    required=False,
    help="Path to the BERT finetuned model. Specify only one model path.",
    default=None,
)
@click.option(
    "--xlnet-base-path",
    "-xlb",
    required=False,
    help="Path to the XLNet base model. Specifiy only one model path.",
    default=None,
)
@click.option(
    "--xlnet-large-path",
    "-xll",
    required=False,
    help="Path to the XLNet large model. Specifiy only one model path.",
    default=None,
)
def serve(
        host,
        port,
        cuda,
        baseline,
        num_cuda_devs=1,
        bert_path=None,
        xlnet_base_path=None,
        xlnet_large_path=None
):
    global MODEL_DICT, DEVICE, BASELINE

    BASELINE = str(baseline)

    if cuda:
        DEVICE = torch.device("cuda:0")
        # will always load inputs on this device even if model parallel option used
    else:
        DEVICE = torch.device("cpu")

    try:
        MODEL_DICT = load_models(DEVICE, num_cuda_devs, bert_path, xlnet_base_path, xlnet_large_path)
    except Exception as e:
        print("An Error occurred: ", e)
        raise e

    app.run(host=host, port=port, debug=True)

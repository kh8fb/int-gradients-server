# int-gradients-server

A cli-based Server for obtaining [intermediate](https://github.com/kh8fb/intermediate-gradients) and [integrated gradients](https://arxiv.org/abs/1703.01365) from given sentences using `curl` requests.

### Installation

This package requires the installation of both this repository as well as [Intermediate Gradients](https://github.com/kh8fb/intermediate-gradients) in an Anaconda environment.

First, create an Anaconda environment:

       conda create -n int-gradients-server

Next, activate the environment, cd into this project's directory and install the requirements with

      conda activate int-gradients-server
      pip install -e .

Finally, cd into the cloned intermediate-gradients direcotry and run

	 pip install -e .

Now your environment is set up and you're ready to go.

### Usage
Activate the server directly from the command line with

	 intgrads -bp /path/to/bert.pth --cpu

This command starts the server and load the model so that it's ready to go when called upon.
The pretrained and finetuned BERT and XLNet models can be downloaded from this [Google drive folder](https://drive.google.com/drive/folders/1KwNZRHwswFu1Nuiz2nvNmBMJ0jnHoA1d?usp=sharing)

You can provide additional arguments such as the hostname, port, and a cuda flag.

After the software has been started, run `curl` with the "model" filepath to get and download the attributions.

      curl http://localhost:8888/model/ -d '{"sequence": "This is the sequence that you want to get the sentiment of"}' --output saved_file.gzip

The gradients are stored in a dictionary with the keys "integrated_gradients", "intermediate_gradients", "step_sizes", and "intermediates".  They are then compressed and able to be retrieved from the saved gzip file with:

      >>> import gzip
      >>> import torch
      >>> from io import BytesIO
      >>> with gzip.open("saved_file.gzip", 'rb') as fobj:
      >>>      x = BytesIO(fobj.read())
      >>>      grad_dict = torch.load(x)




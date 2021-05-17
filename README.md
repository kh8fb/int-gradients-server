# int-gradients-server

A cli-based Server for obtaining [intermediate](https://github.com/kh8fb/intermediate-gradients) and [integrated gradients](https://arxiv.org/abs/1703.01365) from given sentences using `curl` requests.

### Installation

This package requires the installation of both this repository as well as [Intermediate Gradients](https://github.com/kh8fb/intermediate-gradients) in an Anaconda environment.

First, create an Anaconda environment:

       conda create -n int-gradients-server

Next, activate the environment, cd into this project's directory and install the requirements with

      conda activate int-gradients-server
      pip install -e .

Finally, cd into the cloned intermediate-gradients directory and run

	 pip install -e .

Now your environment is set up and you're ready to go.

### Usage
Activate the server directly from the command line with

	 intgrads -bp /path/to/bert.pth --cpu

OR

	intgrads -xlb /path/to/xlnet-base.pth --cuda --num-cuda-devs 2

OR

	intgrads -xll /path/to/xlnet-large.pth --cpu --baseline pad

This command starts the server and load the model so that it's ready to go when called upon.
The pretrained and finetuned BERT and XLNet models can be downloaded from this [Google drive folder](https://drive.google.com/drive/folders/1KwNZRHwswFu1Nuiz2nvNmBMJ0jnHoA1d?usp=sharing)

You can provide additional arguments such as the hostname, port, and a cuda flag.

After the software has been started, run `curl` with the "model" filepath to get and download the attributions.

      curl http://localhost:8888/model/ --data @input_json_file.json --output saved_file.gzip -H "Content-Type:application/json; chartset=utf-8"

      curl http://localhost:8888/model/ -d '{"sequence": "This is the sequence that you want to get the sentiment of"}' --output saved_file.gzip

The gradients are stored in a dictionary with the keys "integrated_gradients", "intermediate_gradients", "step_sizes", and "intermediates".  They are then compressed and able to be retrieved from the saved gzip file with:

      >>> import gzip
      >>> import torch
      >>> from io import BytesIO
      >>> with gzip.open("saved_file.gzip", 'rb') as fobj:
      >>>      x = BytesIO(fobj.read())
      >>>      grad_dict = torch.load(x)


### Running on a remote server
If you want to run int-grads-server on a remote server, you can specify the hostname to be 0.0.0.0 from the command line.  Then use the `hostname` command to find out which IP address the server is running on.

       intgrads -xlb /path/to/xlnet-base.pth -h 0.0.0.0 -p 8008 --cuda --num-cuda-devs 4
       hostname -I
       10.123.45.110 10.222.222.345 10.333.345.678

The first hostname result tells you which address to use in your `curl` request.

    	  curl http://10.123.45.110:8008/model/ --data @input_json_file.json --output saved_file.gzip -H "Content-Type:application/json; chartset=utf-8"

### Baseline options

| --baseline option | Resulting baseline                                                 |
|-------------------|--------------------------------------------------------------------|
| zero              | Tensor of [zero X num_embeddings] leaving the embeddings layer     |
| pad               | Tensor of [pad_id X num_ids] going into the embeddings layer       |
| unk               | Tensor of [unk_id X num_ids] going into the embeddings layer       |
| rand-norm         | Tensor of random normal distribution leaving the embeddings layer  |
| rand-unif         | Tensor of random uniform distribution leaving the embeddings layer |
| period            | Tensor of [period_id X num_ids] going into the embeddings layer    |

### Benchmarking int-gradients server
The models offered by intgrads are often too large to obtain integrated or intermediate gradient results on a single GPU (with large inputs).  As a result, the benchmarking file located at `intgrads/models/run_parallel_models_time_trial.py` can be used to compare the speedup for the parallelized versions of the models.The file runs only the integrated and intermediate gradients (no server) for 1,855 examples from the IMDB dataset that take up the maximum length, 512 tokens.  These incredibly large inputs are difficult for gradient calculations and require models to be distributed between several GPUs.  Below are the average results for running the gradients 10 times over the inputs.  The benchmarking file also has a CLI so that your processing speed can be compared to ours.

| Device | Bert Processing Time | Bert Speedup | XLNet Processing Time | XLNet Speedup |
|--------|----------------------|--------------|-----------------------|---------------|
|   CPU  | 16697 sec            | -            | 57600 sec             | -             |
|  1 GPU | CUDA OUT OF MEMORY   | -            | CUDA OUT OF MEMORY    | -             |
|  2 GPU | 685 sec              | 24.44 x      | CUDA OUT OF MEMORY    | -             |
| 4 GPU  | 671 sec              | 24.83 x      | 1601 sec              | 35.97 x       |
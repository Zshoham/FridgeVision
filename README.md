# Fridge Vision

This is the final assignment for the course Principles of Programming Languages at Ben Gurion University.
> authors: shoahm zarfati - 318501418, hod twito - 315230482

# Goal

We wanted to develop an application that could infer from an image of your fridge what groceries you need to buy.
This is a particularly challenging problem even in object detection because there is not that much data specific for this problem.
Most if not all the data we found was labeled images of groceries in the store or a supermarket.
The difference might not be noticeable, but groceries in your fridge at home often look very different,
if it's because of how they are ordered or in what containers they are placed. We believe
that this problem can be solved using modern machine learning methods and made accessible to everyone.
This will make the process of deciding what you need to put on your grocery list much more comfortable and might
also prevent some issues of forgetting when and what groceries to buy.

# Instructions

Let us start with the project structure, and later we will discuss installation and different usage modes.

## Structure

-  environment config - we provide two files, `requirements.txt` and `conda_env.yml`. These files offer the ability to replicate the development environment used to develop this project.
- `FridgeVision.ipynb` - contains the notebook used to train the classification models and provides an explanation of our results and model selection.
-`frv.py` - is a CLI application meant to be a controller for the project, launch the web app, and make predictions and run the applications in CLI mode.
- `api.py` - is the webserver for the application. It can be launched using the `frv.py` CLI using the `serve` command.
- `webapp` - contains the front end of the web application written in vue.js. When running `serve` for the first time, the front end will be compiled to static files able to be served by flask and put into a new directory `static`. It is also possible to download the application with the compiled front end from the Github Releases.
- `core` - contains the core logic of the application. It is used both by the CLI and the webserver. Inside `core` there are two files:
 	- `detection.py` - here, we can find the code that performs the object detection and all its subtasks like image classification and region selection.
	- `users.py` - here, we can find the database controller for the user's data.
- `test_imgs` - contains a few images usfull for testing the application.
- `.frv` - this is a folder that generates when downloading the model and labels map or running the application, containing the application configuration.
- `doc` - contains some of the documentation for this project, files such as literature review, a presentation, and alike.

## Installation

Because the project relies on machine learning and the TensorFlow library, the project can theoretically be run on the GPU, meaning the models would be much quicker in providing results. This is not guaranteed to be supported through the provided environment files. 
Before moving on, we want to stress that the project works perfectly fine on the CPU apart from slower runtimes.
The GPU configuration is more complicated due to incompatibility between TensorFlow versions and anaconda distributions of those versions.
This project requires Tensorflow version 2.3.* and does not support version 2.1.*. Those requirements and anaconda not properly installing drivers for the 2.3 distribution make the configuration for GPU more difficult.


To install the GPU version, you must first create a conda environment with the following command :
```sh
conda create -n <env name> python=3.7 tensorflow-gpu=2.1 opencv pillow flask
```
next install typer on this environment: 
```sh
conda install -c conda-forge typer 
```
now that the environment is ready with TensorFlow 2.1, you need to update TensorFlow manually using pip: 
```sh
pip install tensorflow-gpu==2.3
``` 
now the environment should be ready to run the project and use the GPU.
Finally, if you wish to recompile the webapp you must have [node.js](https://nodejs.org/en/) and [vue.js](https://cli.vuejs.org/guide/installation.html) installed
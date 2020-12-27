# Implementation

Here we discuss our implementation of the classifier, the region selection we use, and how we combine it all for object detection, in addition to the application itself and its APIs.

## Object Detection

Our classifier is based on the pre-trained [Keras](https://keras.io/api/applications/) EffecientNetB3 network. We also tried training a B0 network, but B3 provided better results at minimal costs.
A more detailed overview of the model implementation is available in the Jupyter notebook, as well as detailed training results that are available 
[here](https://wandb.ai/zshoham/FridgeVision/table?workspace=user-zshoham). Later, for the detection step, as we discussed in the literature review,
we used Selective Search as implemented in [OpenCV](https://docs.opencv.org/4.5.1/d6/d6d/classcv_1_1ximgproc_1_1segmentation_1_1SelectiveSearchSegmentation.html)
in order to create classification regions. Thought we quickly found that selective search created around 20 thousand regions for some images, which is way too much to run on a single GPU, 
not to mention a CPU. In order to decrease the number of regions, we used an algorithm called Non-Maximum Suppression. 
This algorithm removes regions that overlap with each other too much, creating a more manageable number of regions. 
Additionally, we manually eliminated regions that were too small to capture anything meaningful.

Finally, in order to improve accuracy, we also tried to use another variation of Non-Maximum Suppression implemented in 
[tensorflow](https://www.tensorflow.org/api_docs/python/tf/image/non_max_suppression).This algorithm takes into account the prediction probability 
of the classifier when filtering out overlapping bounding boxes, 
which means that only the highest probability detections will stay. This increased performance because the prioritization could take into 
account what the model thinks of the image instead of randomly removing overlapping boxes. On the other hand, this requires running the model on all 
regions produced by Selective Search, making it the slowest method of detection we implemented.


## Application

Our application has two user-facing parts, a CLI and a web UI. The CLI is much more flexible, providing five commands for the user in order to use the application and configure it.
- classify - given an image, runs the classification model on it, and prints the results.
- detect - given an image, runs the whole detection pipeline on the image printing the detected groceries as well as showing the image with bounding boxes drawn on it.
- download-config - downloads a model and label mapping to be used by the application. By default, it downloads the model we trained and the mapping between the numeric classes and the English classes we used.
- run - runs the application with the provided username and image if the username hasn't been seen before printing all the detected groceries. If the username is saved in the database, printing the missing groceries compared to the original image.
- server launches the flask web server and opens a browser to the web app URL.

The web app is much simpler, providing only the functionality available in the run command of the CLI. The app contains a form for inserting the username and uploading an image.
Once the user sends the form, it is received by the flask app, and the detection pipeline is run on it. The server returns the appropriate message to the client describing what 
groceries he is missing or just what groceries appear in the image if it's the first time the username was seen. 
It is important to note that the database is shared between the CLI and the webserver.

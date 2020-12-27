# Literature Review

Here we review some of the work done in object detection and grocery detection in images.
We did not focus on applications used to generate shopping lists, as most of them are either built into fridges
or use simpler methods. In any case, we wanted to focus more on the machine learning side
(which is reflected in our very simple functionalities in terms of shopping list generations).


## Object Detection
The field of object detection is very popular in machine learning as a result of many advancements that were made in the last 5-10 years. 
Object detection is a task where the model is given an image containing several objects and is expected to identify where each object is located in the image (usually by drawing a box around the object) and identify the class of each object (in our case, the class is the type of grocery like milk apple, etc.). 

Object detection is a very similar task to image classification (where each image is given a single class).
With the introduction of [AlexNet](https://proceedings.neurips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html),
proving the effectiveness of CNN networks in image classification, many others followed in different image-related tasks.
The simplest approach to object detection is introduced in [Ross B. Girshick et al.](https://arxiv.org/abs/1311.2524v5) as the R-CNN model. Here the input image is segmented into
many small images. Each of those is fed into an AlexNet classifier, and finally, the AlexNet embeddings are fed into an SVM model for
the final classification for each image segment. With this approach, we can turn any classifier into an object detector by simply adding
this image segmentation at the start. The algorithm used for segmenting the image is called Selective Search. We have used the same approach
only with a different classifier, and we have found two major downsides to this method. First is the runtime because we need to feed all the 
smaller images into the classifier. The classifier needs to do a lot of work that might seem avoidable. The second is the region selection. 
Selective search is relatively good, produces pretty good regions for the classifier, but it obviously cannot consider the data that we are training on,
which makes it very general yet not very good for specific use cases. 

In order to address the issues of runtime and region selection, later methods opted to use "end-to-end" deep neural networks that would be able to predict both the
bounding boxes for the objects and the classes associated with them. This is done by training a network on a dataset where each
image has the coordinates of bounding boxes it contains and the classes for each box. The network is then trained with two outputs, 
a regressor and a classifier. The regressor predicts the coordinates of the bounding boxes, while the classifier predicts the class for each bounding box. 
Finally, a combined loss function is used to guide the training process. This approach means that the network needs to only run once for each image, 
making the network both much faster and able to create more informed decisions on bouncing boxes since it can train on them. 
The architecture described here was introduced in [Fast R-CNN](https://arxiv.org/abs/1504.08083) 
and later improved upon by similar methods such as Faster RCNN YOLO and many more.


## Grocery Detection
Most of the work we found on grocery detection was for detecting groceries in stores and supermarkets.
Even the dataset we used is of supermarket groceries and not fidges. This makes sense because supermarket grocery detection is generally a 
little simpler because of the way the products are separated, and it has a better market as supermarkets can use such systems for keeping track of supply.
One interesting paper we found was [Eran Goldman et al.](https://arxiv.org/abs/1904.00853), where they use an "end-to-end" method similar to Fast RCNN on a very large dataset of annotated images 
of supermarkets with very precise annotation of bounding boxes, up to hundreds per image. This dataset was very impressive but also very 
specialized for supermarkets, in addition to being a little too big for our purposes. Another dataset we considered is [The Freiburg Groceries Dataset](https://arxiv.org/abs/1611.05799). 
The data set was of a manageable size. The paper suggested an approach similar to the original RCNN, and we even tried to train a classifier on it, 
which worked very well. However, the classes in this dataset are not very representative of what you would find in a fridge. We finally settled on a subset of 
[this](https://github.com/marcusklasson/GroceryStoreDataset) data set only titled as "Grocery Store Dataset". We selected 18 classes we thought best represent what you could find in a fridge and trained a classifier on those. The final dataset we created is available on google drive [here](https://drive.google.com/file/d/1yXoDMSPodJb1xBprxplTWy9XwgJm9Mcl/view?usp=sharing)


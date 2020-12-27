import json
import os
from enum import Enum
from typing import Tuple

import cv2
import numpy as np
from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Classifier:

    def __init__(self, model_path: str, label_map_path: str):
        import tensorflow as tf
        from tensorflow.keras.models import load_model

        model = load_model(model_path)
        model.trainable = False
        self.model = tf.keras.Model = tf.keras.Model(model.input, model.output, name="Classifier")

        with open(label_map_path, 'r') as f:
            map_json = f.read()
            f.close()
            self.label_map = json.loads(map_json)

    def predict(self, image: np.ndarray) -> Tuple[str, float]:
        from tensorflow.keras.applications.efficientnet import preprocess_input

        model_input = preprocess_input(np.array([image]))
        prob = self.model.predict(model_input)[0]
        cls = np.argmax(prob)
        return self.label_map[str(int(cls))], prob[cls]

    def predict_all(self, images: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        from tensorflow.keras.applications.efficientnet import preprocess_input

        images = preprocess_input(images)
        prob = self.model.predict(images)
        pred = np.argmax(prob, axis=1)
        return pred, np.array([prob[cls] for cls in pred])


class DetectionQuality(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


def _scale_image(image: Image.Image, scale_to=224) -> Image.Image:
    """
    Scale down an image for better performance when using low quality detection.
    :return: returns a 'image' scaled to 'scale_to' maintaining aspect ratio.
    """
    if image.width > image.height:
        new_width = scale_to
        new_height = int(new_width * image.height / image.width)
    else:
        new_height = scale_to
        new_width = int(new_height * image.width / image.height)

    return image.resize((new_width, new_height), Image.ANTIALIAS)


def generate_regions(image_path, quality: DetectionQuality):
    """
    Generator, first yielding the image that will be used for detection (after augmentation),
    and then yielding all the regions in the format of a tuple - (region, (x1, y1, x2, y2))
    :param image_path: path to image on which the detection should occur
    :param quality: the quality of the detection.
    """
    image = Image.open(image_path)
    if quality == DetectionQuality.low or image.width < 224 or image.height < 224:
        image = _scale_image(image)

    image = np.array(image)

    yield image

    image = image[:, :, ::-1]  # invert RGB to fit cv2 style

    searcher = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    searcher.setBaseImage(image)
    if quality == DetectionQuality.low or quality == DetectionQuality.medium:
        searcher.switchToSelectiveSearchFast()
    else:
        searcher.switchToSelectiveSearchQuality()

    if quality == DetectionQuality.low:
        min_box_size = 5
    else:
        min_box_size = 20

    region_proposals = searcher.process()

    boxes = []
    for (x, y, w, h) in region_proposals:
        if w > min_box_size and h > min_box_size:
            boxes.append((x, y, x + w, y + h))

    regions = _non_max_suppression_fast(np.array(boxes), 0.01)
    for (x1, y1, x2, y2) in regions:
        crop = image[y1:y2, x1:x2, ::-1]  # crop the image and change back to rgb color
        yield crop, (x1, y1, x2, y2)


def hq_region_generator(image_path: str, clf: Classifier):
    """
    Very similar to the generate regions methods, and has the same api, except for
    the addition of the classifier parameter. This region selection process first calculates
    prediction probabilities for each region and then uses non max suppression with the probabilities
    in order to get a better idea of importance to the model.
    :param image_path: path to image on which the detection should occur
    :param clf: a classifier that will be used for generating the region probabilities.
    """
    import tensorflow as tf

    image = Image.open(image_path)
    image = np.array(image)

    yield image

    image = image[:, :, ::-1]  # invert RGB to fit cv2 style

    searcher = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    searcher.setBaseImage(image)
    searcher.switchToSelectiveSearchFast()

    min_box_size = 50

    region_proposals = searcher.process()

    boxes = []
    probs = []
    for (x, y, w, h) in region_proposals:
        if w > min_box_size and h > min_box_size:
            pred, prob = clf.predict(image[y:y + h, x:x + w, ::-1])
            boxes.append((x, y, x + w, y + h))
            probs.append(prob)

    boxes = np.array(boxes)
    boxes = tf.convert_to_tensor(boxes, np.float32)
    probs = np.array(probs)
    probs = tf.convert_to_tensor(probs, np.float32)
    regions = tf.gather(boxes, tf.image.non_max_suppression(boxes, probs, 20)).numpy().astype(np.int32)
    for (x1, y1, x2, y2) in regions:
        crop = image[y1:y2, x1:x2, ::-1]  # crop the image and change back to rgb color
        yield crop, (x1, y1, x2, y2)


def _non_max_suppression_fast(boxes, overlapThresh):
    """
    This code has been taken as is from the blog post - https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
    """
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    # initialize the list of picked indexes
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))
    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")

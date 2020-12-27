import os
import shutil

import requests
import typer
from PIL import Image
import numpy as np
import cv2
import random
import subprocess

import api
from core.detection import DetectionQuality, generate_regions, hq_region_generator
from core.users import Users

app = typer.Typer()

MODEL_URL = "https://api.wandb.ai/files/zshoham/FridgeVision/301yfd2s/model-best.h5"
LABEL_MAP_URL = "https://api.wandb.ai/files/zshoham/FridgeVision/301yfd2s/id-name-map"

DATA_FOLDER_PATH = ".frv"
MODEL_PATH = os.path.join(DATA_FOLDER_PATH, "model.h5")
LABEL_MAP_PATH = os.path.join(DATA_FOLDER_PATH, "label-map.json")


@app.command()
def download_config(model_url: str = typer.Option(MODEL_URL, "--model-url", "-m"),
                    label_map_url: str = typer.Option(LABEL_MAP_URL, "--label-map-url", "-l")):
    """
    Downloads required files for the prediction process to work.
    the two urls point to a website from which the model and label map
    can be downloaded using a simple GET requests.
    """
    if not os.path.isdir(DATA_FOLDER_PATH):
        os.mkdir(DATA_FOLDER_PATH)

    model_data = requests.get(model_url).content
    with open(MODEL_PATH, 'wb') as f:
        f.write(model_data)
        f.close()
        typer.echo("model file successfully downloaded to "
                   + typer.style(MODEL_PATH, fg=typer.colors.GREEN, bold=True))

    label_map_data = requests.get(label_map_url).content
    with open(LABEL_MAP_PATH, 'wb') as f:
        f.write(label_map_data)
        f.close()
        typer.echo("label map file successfully downloaded to "
                   + typer.style(LABEL_MAP_PATH, fg=typer.colors.GREEN, bold=True))


def build_webapp():
    if not os.path.isdir("webapp"):
        typer.echo(typer.style(
            "ERROR: could not find webapp directories required to compile static files",
            fg=typer.colors.RED, bold=True))
        raise typer.Exit(1)
    os.chdir("webapp")
    subprocess.check_call("npm run build", shell=True)
    os.chdir("..")
    shutil.copytree("webapp/dist", "static")


@app.command()
def serve(ctx: typer.Context,
          force: bool = typer.Option(False, "--force", "-f", help="force recompilation of static files if available"),
          model_path: str = typer.Option(MODEL_PATH, "--model-path", "-m"),
          label_map_path: str = typer.Option(LABEL_MAP_PATH, "--label-path", "-l")):
    """
    Launch the webserver and open a browser window on the application running on localhost.
    """

    if ctx.invoked_subcommand is not None:
        return

    if not os.path.isdir("static") or force:
        shutil.rmtree("static", ignore_errors=True)
        build_webapp()

    typer.launch("http://localhost:5000")
    api.serve(DATA_FOLDER_PATH, model_path, label_map_path)


@app.command()
def classify(image_path: str,
             model_path: str = typer.Option(MODEL_PATH, "--model-path", "-m"),
             label_map_path: str = typer.Option(LABEL_MAP_PATH, "--label-path", "-l")):
    """
    Run the model on the given image, predicting what grocery the image contains.
    The two path parameters point to the model and label map that will be used, these should generally
    not be changed.
    """

    from core.detection import Classifier

    classifier = Classifier(model_path, label_map_path)

    img = Image.open(image_path)
    img = np.array(img)

    pred, prob = classifier.predict(img)
    message = "the model predicts that the image class as - "
    message += typer.style(pred, fg=typer.colors.GREEN, bold=True)
    message += " with a probability of - "
    message += typer.style(str(prob), fg=typer.colors.GREEN, bold=True)
    typer.echo(message)


def detect_img(image_path: str, model_path: str, label_map_path: str, quality: DetectionQuality, show_detection: bool):
    from core.detection import Classifier

    classifier = Classifier(model_path, label_map_path)
    groceries = set()

    if quality == DetectionQuality.high:
        region_gen = hq_region_generator(image_path, classifier)
    else:
        region_gen = generate_regions(image_path, quality)

    if show_detection:
        image_copy = next(region_gen).copy()
    else:
        next(region_gen)

    with typer.progressbar(region_gen, length=100, label="Classifying Regions") as progress:
        for (region, (x1, y1, x2, y2)) in progress:
            pred, prob = classifier.predict(region)
            if prob > 0.8:
                groceries.add(pred)
                if show_detection:
                    color = [random.randint(0, 255) for _ in range(0, 3)]
                    cv2.rectangle(image_copy, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(image_copy, pred, (x1 + 5, y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)

    if show_detection:
        Image.fromarray(image_copy).show()

    return list(groceries)


QUALITY_HELP = "Low quality detection uses a lower quality image in addition to generating lower quality regions" \
               "This results in very fast prediction times, but probably not very good detection results." \
               "Medium and High quality detections both use the original image quality, and they are differentiated " \
               "by the quality of the region selection, medium detections use a faster region selection algorithm" \
               "as opposed to high quality detections, which uses a more complex region selection algorithm."


@app.command()
def detect(image_path: str,
           model_path: str = typer.Option(MODEL_PATH, "--model-path", "-m"),
           label_map_path: str = typer.Option(LABEL_MAP_PATH, "--label-path", "-l"),
           quality: DetectionQuality = typer.Option(DetectionQuality.medium, "--quality", "-q",
                                                    help=QUALITY_HELP)):
    """
    Detect what groceries are found in the image. The two path parameters point to the model and label
    map that will be used, these should generally not be changed.
    The quality option controls the quality of region proposals and resulting detections.
    """
    groceries = detect_img(image_path, model_path, label_map_path, quality, True)
    message = "the following groceries were detected: \n"
    for label in groceries:
        message += typer.style(label, fg=typer.colors.GREEN, bold=True) + " "
    typer.echo(message)


@app.command()
def run(ctx: typer.Context,
        username: str = typer.Option(..., "--username", "-u", prompt=True),
        image_path: str = typer.Option(..., "--image-path", "-i", prompt=True),
        model_path: str = typer.Option(MODEL_PATH, "--model-path", "-m"),
        label_map_path: str = typer.Option(LABEL_MAP_PATH, "--label-path", "-l"),
        quality: DetectionQuality = typer.Option(DetectionQuality.medium, "--quality", "-q",
                                                 help=QUALITY_HELP)):

    """
    Run the application predicting what groceries the user needs to buy based on the supplied image.
    The two path parameters point to the model and label
    map that will be used, these should generally not be changed.
    The quality option controls the quality of region proposals and resulting detections.
    """

    if ctx.invoked_subcommand is not None:
        return

    groceries = detect_img(image_path, model_path, label_map_path, quality, False)

    u = Users(DATA_FOLDER_PATH)
    user_groceries = u.load_or_create(username)
    if not user_groceries:
        u.set_user_groceries(username, groceries)
        message = "you are a first time user, from now on we will remember what groceries you need. \n" \
                  "These are the groceries we have detected: \n"
        for label in groceries:
            message += typer.style(label, fg=typer.colors.GREEN, bold=True) + " "
    else:
        message = "you are missing the following groceries: \n"
        for label in [lbl for lbl in user_groceries if lbl not in groceries]:
            message += typer.style(label, fg=typer.colors.GREEN, bold=True) + " "

    typer.echo(message)
    u.close()


if __name__ == "__main__":
    app()

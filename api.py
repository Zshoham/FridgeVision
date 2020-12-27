import os

from flask import Flask, send_from_directory, request

from core.users import Users


def serve(data_folder_path, model_path, label_map_path):
    """
    Initialize flask webserver serving the webapp and exposing a detection API.
    """
    from core.detection import Classifier, generate_regions, DetectionQuality

    classifier = Classifier(model_path, label_map_path)

    app = Flask(__name__, static_url_path='', static_folder='static/')

    temp_image_path = os.path.join(data_folder_path, "tmp_images")

    if not os.path.isdir(temp_image_path):
        os.mkdir(temp_image_path)

    @app.route("/")
    def index():
        return send_from_directory('static', 'index.html')

    @app.route("/detect/<username>", methods=['POST'])
    def detect(username):
        users = Users(data_folder_path)
        # check if the post request has the file part
        if 'file' not in request.files:
            users.close()
            return "no file sent", 400
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            users.close()
            return "no file sent", 400

        image_path = os.path.join(temp_image_path, file.filename)
        file.save(image_path)

        detected_groceries = set()
        quality = DetectionQuality.low
        region_gen = generate_regions(image_path, quality)
        next(region_gen)  # we dont need a copy of the image
        for (region, _) in region_gen:
            pred, prob = classifier.predict(region)
            if prob > 0.8:
                detected_groceries.add(pred)

        os.remove(image_path)  # delete the image after it was processed.
        detected_groceries = list(detected_groceries)
        user_groceries = users.load_or_create(username)
        if not user_groceries:
            users.set_user_groceries(username, detected_groceries)
            message = "you are a first time user, from now on we will remember what groceries you need. \n" \
                      "These are the groceries we have detected: \n"
            for label in detected_groceries:
                message += label + " "
        else:
            missing = [lbl for lbl in user_groceries if lbl not in detected_groceries]
            if len(missing) == 0:
                message = "You are not missing any groceries!"
            else:
                message = "you are missing the following groceries: \n"
                for label in missing:
                    message += label + " "

        users.close()
        return message, 200

    app.run()

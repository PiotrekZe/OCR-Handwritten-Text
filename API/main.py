from flask import Flask, request, jsonify
import numpy as np
import json
import cv2
import os

import segmentation
import inference

app = Flask(__name__)


def convert_to_dict(coords):
    coords_dicts = []
    for coords in coords:
        coords_dicts.append({
            "x1": coords[0],
            "y1": coords[1],
            "x2": coords[2],
            "y2": coords[3]
        })
    return coords_dicts


def create_output(outputs, coords):
    coords_dicts = convert_to_dict(coords)

    decoded_output = []
    for i in range(len(outputs)):
        decoded_output.append({
            'text': outputs[i],
            'coords': coords_dicts[i]
        })
    # return json.dumps({'decoded outputs': decoded_output})
    return jsonify({'decoded outputs': decoded_output})


# @app.route('/')


@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return 'No image uploaded', 400

    image_file = request.files['image']

    json_data = request.form.get('json')
    bounding_boxes = json.loads(json_data)['boundingBoxes']
    print(os.getcwd())
    print(os.listdir(os.getcwd()))
    print(os.listdir("./"))
    print("jakas zmiana zeby było widac")
    # You can now use image_file.save() to save the image to disk
    # Or perform any processing you need on the image
    # img = Image.open(image_file)

    image_data = image_file.read()
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

    num_channels = img.shape[2] if len(img.shape) == 3 else 1
    if num_channels == 1:
        grayscale_image = img
    else:
        grayscale_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # img = cv2.imread(image_file)

    _, binary_image = cv2.threshold(grayscale_image, 150, 255, cv2.THRESH_BINARY_INV)
    # binary_image = cv2.adaptiveThreshold(grayscale_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 4)
    # cv2.imwrite("C:/Users/Piotr/Desktop/IAM_img/imageimage3.png", binary_image)
    # print(np.array(img).shape)
    # print(np.array(grayscale_image).shape)

    segmentated_image = segmentation.segment_image(binary_image)
    # print("segmentated_image", segmentated_image.shape)
    # cv2.imwrite("C:/Users/Piotr/Desktop/IAM_img/imageimage4.png", segmentated_image)

    contours, _ = cv2.findContours(segmentated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    extracted_lines, extracted_contours, extracted_outputs = [], [], []

    for i, contour in enumerate(contours):
        contour_mask = np.zeros_like(segmentated_image)
        cv2.drawContours(contour_mask, [contour], 0, 255, -1)
        text_line = cv2.bitwise_and(binary_image, binary_image, mask=contour_mask)
        x, y, w, h = cv2.boundingRect(contour)
        text_line_cropped = text_line[y:y + h, x:x + w]
        if cv2.contourArea(contour) >= 1500:
            extracted_lines.append(text_line_cropped)
            extracted_contours.append([x, y, x + w, y + h])
            output = inference.call_model(text_line_cropped)
            extracted_outputs.append(output)
            # print(output)
            # print([x, y, x + w, y + h])

    json_output = create_output(extracted_outputs, extracted_contours)
    # print(json_output)
    # print(type(json_output))
    # print(extracted_outputs)
    # print(extracted_contours)
    # output = inference.call_model(segmentated_image)
    # image = Image.open(io.BytesIO(image_file.read())).convert("RGB")
    # print(np.array(image).shape)
    # print(bounding_boxes)
    # print(type(bounding_boxes))
    # plt.imsave("C:/Users/Piotr/Desktop/IAM_img/imageimage2.png", image)
    #
    # segmentated_image = segmentation.segment_image("temp_path", image)
    # inference.call_model(segmentated_image)
    # print(segmentated_image.shape)

    return json_output, 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

'''
# Do zrobienia:

1. Ujednolicić przy przekazywnaiu zdjęć - segmentacja musi wiedzieć co dostanie na wejściu i tyle,
umówiliśmy się na jeden kanał to teraz cierpimy i musi być jeden kanał w szarości. 
2. Na ten moment wszystko działa - segmentacja działa i zwraca, OCR działa i zwraca
3. Musimy po ujednoliceniu wczytywania zadbać o wysyłanie fragmentami te czesci po segmentacji do OCR i zapisywanie
coordynatów


WAZNE
wczytywane zdjęcie nie moze mieć 4 kanałów bo się wszystko pierdoli i przestaje działać - trzeba to sprawdzać i rzucać wyjątkiem=
'''

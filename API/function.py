import numpy as np
from PIL import Image
import requests
import json


def fun():
    for i in range(4):
        print(i)


API_URL = "http://localhost:5000/echo"


def test_predict_with_file():
    image_file = Image.open("C:/Users/Piotr/Desktop/a01-014u1.png")
    image_file = np.array(image_file)

    print("jehehehehe")
    response = requests.post(API_URL, files={"file": "image_file"})
    if response.status_code == 201:
        print("dupsko dupsko")
        response_data = response.json()
        print(response_data['message'])
    else:
        assert response.status_code == 201
        print("Error in API")


def kupa():
    p = "D:/Databases/06/forms_obciete/a01-000u.png"  # ja to zapisałem tak jak powinienem, ale później jest z tym
    # problem bo nie jestem konskwentny w stosowaniu tej inwersji gdzie biały jest tekst, a ciemne tło
    p = "C:/Users/Piotr/Desktop/IAM_img/a01-000u.png"
    # p = "C:/Users/Piotr/Desktop/IAM_img/imageimage3.png"
    # p = "C:/Users/Piotr/Desktop/IAM_img/a01-049.png"
    # p = "C:/Users/Piotr/Desktop/IAM_img/linie1.jpg"

    with open(p, 'rb') as f:
        img = f.read()
    api_link = "http://localhost:5000/upload_image"

    bounding_boxes_data = {
        "boundingBoxes": [
            {
                "coords": {
                    "x1": 0,
                    "y1": 0,
                    "x2": 100,
                    "y2": 100
                }
            },
            {
                "coords": {
                    "x1": 150,
                    "y1": 50,
                    "x2": 200,
                    "y2": 150
                }
            }
        ]
    }

    file = {'image': img}
    data = {'json': json.dumps(bounding_boxes_data)}
    response = requests.post(api_link, files=file, data=data)
    print(response.json())


def main():
    # test_predict_with_file()
    kupa()
    print("ejjejejejej")


if __name__ == "__main__":
    main()

import requests
import cv2
# import base64

path = ('/Users/duongphamminhdung/Downloads/z5449299028482_9cfbfd102c882eb6ac5aec7ada4307c3.jpg')
img = cv2.imread(path)
ret, img_bytes = cv2.imencode('.jpg', img)
url = 'http://127.0.0.1:8000/facerfb320'
file = {
        'filename': 'a.jpg',
        'content_type': 'image/jpg',
        'file':  img_bytes,
       # 'file': open(path, 'rb')
}

resp = requests.post(url=url, files=file)
import ipdb; ipdb.set_trace()
print(resp.json())
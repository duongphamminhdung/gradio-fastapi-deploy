import requests
import json
import cv2
import numpy as np

url = 'http://127.0.0.1:8000/facerfb320'
import numpy as np
import cv2
import base64

img = cv2.imread('/Users/duongphamminhdung/Downloads/z5449299028482_9cfbfd102c882eb6ac5aec7ada4307c3.jpg')
ret, image_bytes = cv2.imencode('.jpg', img)
img_bytes = base64.b64encode(image_bytes)
img_encode = img_bytes.decode('utf-8')
file = {
        'filename': 'a',
        'content_type': 'text/plain',
        'file':  img_encode,
       # 'file': open(path, 'rb')
}

r = requests.post(url=url, files=file)
import ipdb; ipdb.set_trace()
# print(r.json())
img_bytes = base64.b64decode(r.content)
img = cv2.imdecode(np.asarray(bytearray(img_bytes), dtype=img.dtype), cv2.IMREAD_UNCHANGED)

cv2.imwrite('image/d.jpg', img)
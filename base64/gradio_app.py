import gradio as gr
import requests
import json
import numpy as np
import cv2
import base64

url = 'http://127.0.0.1:8000/facerfb320'

def send_request(token, img):
        ret, img_bytes = cv2.imencode('.jpg', img)
        img_bytes = base64.b64encode(img_bytes)
        img_encode = img_bytes.decode('utf-8')

        file = {
        'filename': 'a',
        'content_type': 'text/plain',
        'file':  img_encode,
       # 'file': open(path, 'rb')
        }       

        r = requests.post(url=url, files=file)
        # import ipdb; ipdb.set_trace()
        # img_detect = cv2.imdecode(np.frombuffer(r.content, dtype=img_bytes.dtype), cv2.IMREAD_UNCHANGED)
        img_bytes = base64.b64decode(r.content)
        img = cv2.imdecode(np.asarray(bytearray(img_bytes), dtype=img.dtype), cv2.IMREAD_UNCHANGED)
        return token, img

demo = gr.Interface(send_request, inputs=["text" , gr.Image(type='numpy')], outputs=['text', gr.Image(type='numpy')])
if __name__ == "__main__":
    demo.launch()
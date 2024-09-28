import gradio as gr
import requests
import json
import numpy as np
import cv2

url = 'http://127.0.0.1:8000/facerfb320'

def send_request(token, img):
        ret, img_bytes = cv2.imencode('.jpg', img)
        file = {
                'filename': 'a.jpg',
                'content_type': 'image/jpg',
                'file':  img_bytes,
                # 'file': open(path, 'rb')
        }

        r = requests.post(url=url, files=file)
        # import ipdb; ipdb.set_trace()
        img_detect = cv2.imdecode(np.frombuffer(r.content, dtype=img_bytes.dtype), cv2.IMREAD_UNCHANGED)
        return token, img_detect

demo = gr.Interface(send_request, inputs=["text" , gr.Image(type='numpy')], outputs=['text', 'image'])
if __name__ == "__main__":
    demo.launch()
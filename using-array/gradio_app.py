import gradio as gr
import requests
import json
import numpy as np
import base64

url = 'http://127.0.0.1:8000/facerfb320'
headers = {"Content-Type": "application/json"}

def send_request(token, img):
        img_list = img.tolist()
        img_dtype = img.dtype
        img_shape = img.shape
        data = {
                'token': 'hehehe',
                'image': img_list,
                'dtype': str(img_dtype),
                'shape': img_shape,
        }

        r = requests.get(url, data=json.dumps(data), headers=headers)
        img = np.array(r.json()['img'])
        img_detect = np.reshape(img.astype(dtype=data['dtype']), data['shape'])
        return token, img_detect

demo = gr.Interface(send_request, inputs=["text" , gr.Image(type='numpy')], outputs=['text', gr.Image(type='numpy')])
if __name__ == "__main__":
    demo.launch()

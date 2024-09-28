from typing import Union
import numpy as np
from fastapi import FastAPI, Form
from pydantic import BaseModel
import base64
from face2stk_rfb320 import detect

app = FastAPI()
TOKEN = 'hehehe'

class Image(BaseModel):
    token: str
    image: list
    dtype: str
    shape: tuple


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/facerfb320")
def read_item(src_img: Image):
    # import ipdb; ipdb.set_trace()
    # print(req)
    # print(token, image)
    if src_img.token != TOKEN:
        return {
            'status': 'fail: wrong or none token ',
            'img': None
        }
    else :
        img_bytes = np.array(src_img.image)
        img = np.reshape(img_bytes.astype(src_img.dtype), src_img.shape)
        img_detect = detect(img)
        # img_detect_bytes = base64.b64encode(img_detect)
        img_detect_list = img_detect.tolist()
    return {'status':'sucess', 'img': img_detect_list}
    


# @app.put("/items/{item_id}")
# def update_item(item_id: int, item: Item):
#     return {"item_name": item.name, "item_id": item_id}
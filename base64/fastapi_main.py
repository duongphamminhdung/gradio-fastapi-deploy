
import numpy as np
from fastapi import FastAPI, Form, File, UploadFile, Response
from fastapi.responses import FileResponse

import io
from face2stk_rfb320 import detect
import cv2
import base64

app = FastAPI()
TOKEN = 'hehehe'

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/facerfb320")
def upload(file: UploadFile = File(...)):
    try:
        # import ipdb; ipdb.set_trace()
        contents = file.file.read()
        img = np.asarray(bytearray(base64.b64decode(contents)), dtype=np.uint8)
        img_decoded = cv2.imdecode(img, cv2.IMREAD_UNCHANGED)
        img_detect = detect(img_decoded)
        ret, img_bytes = cv2.imencode('.jpg', img_detect)
    except Exception as e:
        print(e)
        return {"message": "There was an error uploading the file "}
    finally:
        file.file.close()

    # return {"message": f"Successfully uploaded {file.filename}"}
    return base64.b64encode(img_bytes).decode('utf-8')
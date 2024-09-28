from PIL import Image
from openvino.runtime import Core
import numpy as np
import cv2
from model import utils

class FaceDetector:
    def __init__(self,
                 model,
                 confidence_thr=0.5,
                 overlap_thr=0.7):
        # load and compile the model
        core = Core()
        model = core.read_model(model=model)
        compiled_model = core.compile_model(model=model)
        self.model = compiled_model

        # 'Cause that model has more than one output,
        # We are saving the names in a more human frendly
        # variable to remember later how to recover the output we wish
        # In our case here, is a output for hte bbox and other for the score
        # /confidence. Have a look at the openvino documentation for more i
        self.output_scores_layer = self.model.output(0)
        self.output_boxes_layer  = self.model.output(1)
        # confidence threshold
        self.confidence_thr = confidence_thr
        # threshold for the nonmaximum suppression
        self.overlap_thr = overlap_thr

    def preprocess(self, image):
        """
            input image is a numpy array image representation,
            in the BGR format of any shape.
        """
        # resize to match the expected by the model
        input_image = cv2.resize(image, dsize=(320,240))
        # changing from [H, W, C] to [C, H, W]. "channels first"
        input_image = np.expand_dims(input_image.transpose(2,0,1), axis=0)
        return input_image

    def posprocess(self, pred_scores, pred_boxes, image_shape):
        # get all predictions with more than confidence_thr of confidence
        filtered_indexes = np.argwhere( pred_scores[0,:,1] > self.confidence_thr  ).tolist()
        filtered_boxes   = pred_boxes[0,filtered_indexes,:]
        filtered_scores  = pred_scores[0,filtered_indexes,1]

        if len(filtered_scores) == 0:
            return [],[]

        # convert all boxes to image coordinates
        h, w = image_shape
        def _convert_bbox_format(*args):
            bbox = args[0]
            x_min, y_min, x_max, y_max = bbox
            x_min = int(w*x_min)
            y_min = int(h*y_min)
            x_max = int(w*x_max)
            y_max = int(h*y_max)
            return x_min, y_min, x_max, y_max

        bboxes_image_coord = np.apply_along_axis(_convert_bbox_format, axis = 2, arr=filtered_boxes)

        # apply non-maximum supressions
        bboxes_image_coord, indexes = utils.non_max_suppression(bboxes_image_coord.reshape([-1,4]),
                                                                overlapThresh=self.overlap_thr)
        filtered_scores = filtered_scores[indexes]
        return bboxes_image_coord, filtered_scores

    def inference(self, image):
        input_image = self.preprocess(image)
        # inference
        pred_scores = self.model( [input_image] )[self.output_scores_layer]
        pred_boxes = self.model( [input_image] )[self.output_boxes_layer]

        image_shape = image.shape[:2]
        faces, scores = self.posprocess(pred_scores, pred_boxes, image_shape)
        return faces, scores
# def create_transparent_image(image, threshold, mode='equal', get_full_object=False):

def detect(img):
    detector = FaceDetector('model/public/ultra-lightweight-face-detection-rfb-320/FP16/ultra-lightweight-face-detection-rfb-320.xml')
    bboxes, scores = detector.inference(img)
    if len(bboxes) > 0:
        for box in bboxes:
            x1, y1, x2, y2 = box[0]-5, box[1], box[2], box[3]
            img = cv2.rectangle(img, (x1, y1), (x2, y2), (100, 100, 100), 2)
            # img = np.array(img_raw_pil)
            # print(img.shape)
            # cv2.imwrite(os.path.join('/Users/duongphamminhdung/Documents/twitter/clths/', name, day, os.path.basename(i)), img)

    return img
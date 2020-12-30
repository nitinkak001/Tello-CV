import cv2
import tensorflow as tf
from tf_bodypix.api import download_model, load_model, BodyPixModelPaths
from PIL import Image
import time


class Tello_CV:

    def __init__(self):
        self.scale_percent = 30
        self.bodypix_model = load_model(download_model(
            BodyPixModelPaths.MOBILENET_FLOAT_50_STRIDE_16
        ))

    def segment_body_part(self, frame, part_names):
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image_array = tf.keras.preprocessing.image.img_to_array(pil_img)
        result = self.bodypix_model.predict_single(image_array)
        mask = result.get_mask(threshold=0.92)
        part_mask = result.get_part_mask(mask, part_names=part_names)
        return part_mask.astype('int16')

    def get_mask_centroid(self, mask):
        M = cv2.moments(mask)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX = -1
            cY = -1
        return cX, cY

    def draw_circle(self, cX, cY, img):
        cv2.circle(img, (cX, cY), 5, (255, 255, 255), -1)
        cv2.putText(img, "centroid", (cX - 25, cY - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        return img

    def preprocess_frame(self, img):
        width = int(img.shape[1] * self.scale_percent / 100)
        height = int(img.shape[0] * self.scale_percent / 100)
        dim = (width, height)
        return cv2.resize(img, dsize=dim)

    def detect(self, frame):
        cX, cY = self.get_mask_centroid(
            self.segment_body_part(frame, ['left_face', 'right_face']))
        points = []
        points.append((cX, cY))
        if cX > 0:
            draw_skeleton_flag = True
        else:
            draw_skeleton_flag = False
        cmd = "stay"
        return cmd, draw_skeleton_flag, points

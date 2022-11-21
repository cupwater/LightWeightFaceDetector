'''
Author: Peng Bo
Date: 2022-11-13 22:19:55
LastEditTime: 2022-11-21 09:30:56
Description: 

'''

import time
import cv2
import numpy as np
import onnxruntime as ort
import pdb


def detect_face_lms(src_image, ort_session, input_size=(320, 180)):    
    input_name = ort_session.get_inputs()[0].name
    def _preprocess(src_image):
        # pre-process the input image 
        input_data = cv2.cvtColor(src_image, cv2.COLOR_BGR2RGB)
        input_data = cv2.resize(input_data, input_size)
        input_data = (input_data - np.array([104, 117, 123]))
        target_data = np.expand_dims(np.transpose(input_data, [2, 0, 1]), axis=0)
        target_data = target_data.astype(np.float32)
        return target_data

    input_data = _preprocess(src_image)
    start_time = time.time()
    loc, conf, landms = ort_session.run(None, {input_name: input_data})
    print("inference time:{}".format(time.time() - start_time))
    return loc, conf, landms

if __name__ == '__main__':
    onnx_path = "weights/mbnv3_320x180.onnx"
    ort_session = ort.InferenceSession(onnx_path)
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        loc, conf, landms = detect_face_lms(frame, ort_session)
        # pdb.set_trace()
        cv2.imshow("Result", frame)
        key = cv2.waitKey(-1)
        if key==27 or key == ord("q"):
            exit(0)
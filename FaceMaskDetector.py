from __future__ import absolute_import

import os

from keras.models import load_model
import cv2
import numpy as np


class FaceMaskDetector(object):
    def __init__(self, model=None, alert_config=None):

        self.__path = os.path.dirname(os.path.realpath(__file__))

        model_path = model if model else os.path.join(self.__path,
                                                      '/Model/Fmd.h5')
        classifier_path = os.path.join(
            self.__path, 'Haarcascades/haarcascade_frontalface_default.xml')

        self.model = load_model(self.model_path)
        self.face_detect_classifier = cv2.CascadeClassifier(classifier_path)

        self.video_source = cv2.VideoCapture(0)

        self.txt_results = {0: 'Wearing Mask', 1: 'No Mask'}
        self.colors = {0: (0, 255, 0), 1: (0, 0, 255)}

    def start(self):

        while True:

            ret, img = self.video_source.read()
            img_grayscaled = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_detect_classifier.detectMultiScale(
                img_grayscaled, 1.3, 5)

            for (x, y, w, h) in faces:
                face_img = img_grayscaled[y:y + w, x:x + w]
                normalized_img = cv2.resize(face_img, (112, 112)) / 255.0
                reshaped_img = np.reshape(normalized_img, (1, 112, 112, 1))

                result = self.model.predict(reshaped_img)
                label = np.argmax(result, axis=1)[0]

                cv2.rectangle(img, (x, y), (x + w, y + h), self.colors[label],
                              2)
                cv2.rectangle(img, (x, y - 40), (x + w, y + h),
                              self.colors[label], -1)
                cv2.putText(img, self.txt_results[label], (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

                # If no mask and alert is enabled
                if label == 1:
                    # send mail
                    print("No Mask")
                else:
                    pass
                    break

            cv2.imshow('FaceMaskDetector Live Feed')
            # Press Esc Key to Exit
            key = cv2.waitKey(1)
            if key == 27: break

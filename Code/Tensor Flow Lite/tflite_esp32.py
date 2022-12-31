# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Main script to run the object detection routine."""

#Credit to TensorFlow Authors & FREEDOMTECH for the basis of the code.
#Additional modifications made by the ECE4300 Group C Team

import argparse
import sys
import time

import cv2
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
import utils
from esp32tst import cam


def run(model: str, camera_id: int, width: int, height: int, num_threads: int,
        enable_edgetpu: bool) -> None:

    """Continuously run inference on images acquired from the camera.

      Args:
        model: Name of the TFLite object detection model.
        camera_id: The camera id to be passed to OpenCV.
        width: The width of the frame captured from the camera.
        height: The height of the frame captured from the camera.
        num_threads: The number of CPU threads to run the model.
        enable_edgetpu: True/False whether the model is a EdgeTPU model.
    """

    # Variables to calculate FPS
    start_time = time.time()

    # Visualization parameters
    row_size = 20  # pixels
    left_margin = 24  # pixels
    text_color = (0, 0, 255)  # red
    font_size = 1
    font_thickness = 1
    fps_avg_frame_count = 10

    # Initialize the object detection model
    base_options = core.BaseOptions(
      file_name=model, use_coral=enable_edgetpu, num_threads=num_threads)
    detection_options = processor.DetectionOptions(
      max_results=3, score_threshold=0.3)
    options = vision.ObjectDetectorOptions(
      base_options=base_options, detection_options=detection_options)
    detector = vision.ObjectDetector.create_from_options(options)
    
    start_time = time.time()
    last_process = time.time()
    InferenceTime = round(time.time()-last_process, 2)
    currentTime = round(time.time()-start_time,2)
        
    while True:
        print(currentTime, ',', InferenceTime)
        
        #retrieve image and resize to (640x480)
        image = cv2.resize(cam(),(640,480))
        #rotate image
        image = cv2.rotate(image, cv2.cv2.ROTATE_180)

        # Convert the image from BGR to RGB as required by the TFLite model.
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Create a TensorImage object from the RGB image.
        input_tensor = vision.TensorImage.create_from_array(rgb_image)

        # Run object detection estimation using the model.
        detection_result = detector.detect(input_tensor)

        # Draw keypoints and edges on input image
        image = utils.visualize(image, detection_result)

        # Calculate the Inference time
        InferenceTime = round(1000 * (time.time()-last_process), 2)
        currentTime = round(time.time()-start_time,2)
        last_process = time.time()

        # Stop the program if the ESC key is pressed.
        if cv2.waitKey(1) == 27:
          break

        cv2.imshow('ESPCAM32 Tensorflow Lite Object Detection', image)

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
      '--model',
      help='Path of the object detection model.',
      required=False,
      default='efficientdet_lite0.tflite')
    parser.add_argument(
      '--cameraId', help='Id of camera.', required=False, type=int, default=0)
    parser.add_argument(
      '--frameWidth',
      help='Width of frame to capture from camera.',
      required=False,
      type=int,
      default=640)
    parser.add_argument(
      '--frameHeight',
      help='Height of frame to capture from camera.',
      required=False,
      type=int,
      default=480)
    parser.add_argument(
      '--numThreads',
      help='Number of CPU threads to run the model.',
      required=False,
      type=int,
      default=4)
    parser.add_argument(
      '--enableEdgeTPU',
      help='Whether to run the model on EdgeTPU.',
      action='store_true',
      required=False,
      default=False)
    args = parser.parse_args()

    run(args.model, int(args.cameraId), args.frameWidth, args.frameHeight,
      int(args.numThreads), bool(args.enableEdgeTPU))

if __name__ == '__main__':
    main()

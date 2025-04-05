import numpy as np
import rospy
from sklearn.linear_model import RANSACRegressor
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import cv2 as cv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Model Loading")
processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(device)

ocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
ocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten").to(device)

print("Model Loaded")
text_labels = [["digit on the wall"]]

padding = 8


def recognize(image, detection_threshold=0.0, area_threshold=400.0, debug=False, return_max=False):
    inputs = processor(text=text_labels, images=image, return_tensors="pt").to(device)
    outputs = model(**inputs)
    target_sizes = torch.Tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs=outputs,
                                                      target_sizes=target_sizes,
                                                      threshold=detection_threshold)[0]
    image_array = np.array(image)
    digit_and_position = []
    max_area, max_result = -1, None
    for box, score, labels in zip(results["boxes"], results["scores"], results["labels"]):
        x1, y1, x2, y2 = [int(x) for x in box.tolist()]
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        if area < area_threshold:
            continue
        cv.rectangle(image_array, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # recognize the digit
        processed_image = ocr_processor(
            images=Image.fromarray(image_array[y1 - padding:y2 + padding, x1 - padding:x2 + padding]),
            return_tensors="pt").to(device)
        pixel_values = processed_image.pixel_values
        generated_ids = ocr_model.generate(pixel_values, max_new_tokens=1)
        generated_text = ocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        if generated_text.isdigit() and len(generated_text) == 1:
            digit_and_position.append([int(generated_text), (x1, y1, x2, y2)])
            print(digit_and_position[-1])
            if area > max_area:
                max_area = area
                max_result = digit_and_position[-1]

    if len(digit_and_position) > 0 and debug:
        cv.imshow("image", image_array)
        cv.waitKey(0)

    if not return_max:
        return digit_and_position
    else:
        return max_result


def get_box_position(bbox, disparity, baseline, camera_info, box_size=0.8):
    x1, y1, x2, y2 = bbox

    K = camera_info.K
    fx, fy = K[0], K[4]
    cx, cy = K[2], K[5]
    center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
    if disparity[center_y][center_x] == 0:
        rospy.logerr("Do not get the depth of center point")
        return None
    z = (fx * baseline) / disparity[center_y][center_x]
    plane_center = ((center_x - cx) / fx, (center_y - cy) / fy, z)

    points = []
    for u in range(x1, x2 + 1):
        for v in range(y1, y2 + 1):
            try:
                if disparity[v][u] == -1:
                    continue
            except IndexError:
                print(u, v)
                print(type(u), type(u))
            z = (fx * baseline) / disparity[v][u]
            x = (u - cx) / fx
            y = (v - cy) / fy
            points.append([x, y, z])

    if len(points) < 3:
        rospy.logerr(f"Points({len(points)}) not adequate for plane fitting")
        return None

    points = np.array(points)

    # fit a plane
    X = points[:, :2]
    v = points[:, 2]
    ransac = RANSACRegressor()
    ransac.fit(X, v)

    # plane equation: z = a*x + b*y + c
    a, b = ransac.estimator_.coef_
    c = ransac.estimator_.intercept_
    normal_vector = np.array([a, b, -1])
    normal_vector = normal_vector / np.linalg.norm(normal_vector)

    dot = np.dot(plane_center[:2], normal_vector[:2])
    if dot < 0:
        box_center = plane_center + (box_size / 2) * normal_vector
        print(f"box center (x, y): {box_center[:2]}")
    elif dot > 0:
        box_center = plane_center - (box_size / 2) * normal_vector
        print(f"box center (x, y): {box_center[:2]}")
    else:
        print("Should not happen")
        box_center = None
    # return box_center
    return plane_center

import numpy as np
import rospy
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import cv2 as cv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

rospy.loginfo("Model Loading")
processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(device)

ocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
ocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten").to(device)

rospy.loginfo("Model Loaded")
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
            if generated_text == "0":
                generated_text = 1
            digit_and_position.append([int(generated_text), (x1, y1, x2, y2)])
            rospy.loginfo(digit_and_position[-1])
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
    if disparity[center_y][center_x] != -1:
        z = (fx * baseline) / disparity[center_y][center_x]
        plane_center = ((center_x - cx) * z / fx, (center_y - cy) * z / fy, z)
        return plane_center
    rospy.logwarn("Do not get the depth of center point")
    half_width, half_height = (x2 - x1) // 2, (y2 - y1) // 2
    for dx in range(-half_width, 0):
        for dy in range(-half_height, 0):
            x1, y1 = center_x + dx, center_y + dy
            x2, y2 = center_x - dx, center_y - dy
            if disparity[y1][x1] != -1 and disparity[y2][x2] != -1:
                z1_3d = (fx * baseline) / disparity[y1][x1]
                z2_3d = (fx * baseline) / disparity[y2][x2]
                x1_3d = (x1 - cx) * z1_3d / fx
                x2_3d = (x2 - cy) * z2_3d / fy
                y1_3d = (y1 - cy) * z1_3d / fy
                y2_3d = (y2 - cy) * z2_3d / fy
                return (x1_3d + x2_3d) / 2, (y1_3d + y2_3d) / 2, (z1_3d + z2_3d) / 2
    rospy.logerr("Could not find a symmetric depth")
    return None

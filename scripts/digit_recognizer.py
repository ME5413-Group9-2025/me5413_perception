import numpy as np
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

padding = 5


def recognize(image, detection_threshold=0.0, area_threshold=400.0, debug=False):
    inputs = processor(text=text_labels, images=image, return_tensors="pt").to(device)
    outputs = model(**inputs)
    target_sizes = torch.Tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs=outputs,
                                                      target_sizes=target_sizes,
                                                      threshold=detection_threshold)[0]
    image_array = np.array(image)
    digit_and_position = []
    for box, score, labels in zip(results["boxes"], results["scores"], results["labels"]):
        x1, y1, x2, y2 = [int(x) for x in box.tolist()]
        if (x2 - x1) * (y2 - y1) < area_threshold:
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
            digit_and_position.append([int(generated_text), (int((x1 + x2) / 2), int((y1 + y2) / 2))])
            print(digit_and_position[-1])

    if len(digit_and_position) > 0 and debug:
        cv.imshow("image", image_array)
        cv.waitKey(0)
    return digit_and_position

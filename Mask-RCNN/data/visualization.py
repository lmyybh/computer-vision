import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage

def PIL2CV(image):
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

def show_image(image, target):
    img = PIL2CV(ToPILImage()(image))
    bboxes = target.get_field("bboxes").convert("xyxy").bbox.numpy().astype(np.int32)
    masks = target.get_field("masks").mask.numpy()
    masks[masks>=0.5] = 1
    masks[masks<0.5] = 0
    masks = masks.astype(np.uint8)
    for i in range(bboxes.shape[0]):
        img = cv2.rectangle(img, bboxes[i][:2], bboxes[i][2:], (0, 255, 0), 1)

        contours, hierarchy = cv2.findContours(masks[i], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        maskImg = np.zeros_like(img)
        color = tuple(int(x) for x in np.random.uniform(low=0, high=255, size=3))
        maskImg = cv2.drawContours(maskImg, contours, -1, color, -1)
        img = cv2.add(img, (maskImg * 0.8).astype(np.uint8))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 8))
    plt.imshow(img)
    plt.show()
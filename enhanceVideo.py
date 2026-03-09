import cv2 as cv
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

cap=cv.VideoCapture(1)
enhancers = {'Sharpness':1,'Contrast':1,'Brightness':1,'Color':1}
adjustments = {
            ord('S'): ('Sharpness', 0.1),
            ord('s'): ('Sharpness', -0.1),
            ord('T') : ('Contrast', 0.1),
            ord('t') : ('Contrast', -0.1),
            ord('B'): ('Brightness', 0.1),
            ord('b'): ('Brightness', -0.1),
            ord('C'): ('Color', 0.1),
            ord('c'): ('Color', -0.1),
        }

while True:
    ret,frame=cap.read()
    if ret:
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frame_rgb=cv.flip(frame_rgb,1)
        image = Image.fromarray(frame_rgb)
        # image = image.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
        # image = image.filter(ImageFilter.EDGE_ENHANCE)
        # image = ImageOps.autocontrast(image, cutoff=2)
            
        for enhancer_type, factor in enhancers.items():
            enhancer = getattr(ImageEnhance, enhancer_type)(image)
            image = enhancer.enhance(factor)
        
        enhanced_frame = cv.cvtColor(np.array(image), cv.COLOR_RGB2BGR)
        cv.imshow('Enhanced Video', enhanced_frame)#cv.resize(enhanced_frame,(640,360)))

        key = cv.waitKey(1) & 0xFF
        if key in adjustments:
            setting, change = adjustments[key]
            current_value = enhancers[setting]
            new_value = current_value + change#max(0.1, current_value + change)
            enhancers[setting] = new_value
            print(f"Adjusted {setting} to {enhancers[setting]:.1f}")

        if key==ord('q'):
             break
        
cap.release()
cv.destroyAllWindows()       

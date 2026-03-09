import cv2
import easyocr
from PIL import Image
# Image.open('image.jpg').verify()

# Initialize EasyOCR reader (run only once)
reader = easyocr.Reader(['en'])  # Specify languagese

# result = reader.readtext('Screenshot 2025-04-27 194918.png')
# for detection in result:
#     print(detection[1])

cap = cv2.VideoCapture(0)
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.GaussianBlur(frame,(3,3),1)
        # Perform OCR
        results = reader.readtext(frame)
        
        # Display results
        for (bbox, text, prob) in results:
            # Draw bounding box
            (tl, tr, br, bl) = bbox
            tl = (int(tl[0]), int(tl[1]))
            br = (int(br[0]), int(br[1]))
            cv2.rectangle(frame, tl, br, (0, 255, 0), 2)
            print(text)
            # Put text
            cv2.putText(frame, text, (tl[0], tl[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 10), 2)

        cv2.imshow("Live OCR", frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()


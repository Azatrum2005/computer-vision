import cv2
import numpy as np

def empty(a):
    pass

def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

def getContours(img,imgContour):
    contours,hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        peri = cv2.arcLength(cnt,True)
        if peri>180 and 10000<=area<=250000:
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 4)
            radius=peri/6.28
            # print("radius:",peri/6.28)
            approx = cv2.approxPolyDP(cnt,0.001*peri,True)
            # objCor = len(approx)
            x, y, w, h = cv2.boundingRect(approx)
            # cv2.rectangle(imgContour,(x,y),(x+w,y+h),(0,255,0),4)
            mx=x+w//2
            my=y+h//2
            cv2.rectangle(imgContour,(mx-150,my-180),(mx+150,my+120),(0,255,0),4)
            try:
                cv2.imshow("face",imgContour[my-180:my+120,mx-150:mx+150])
            except:
                continue

# path = 'Resources/lambo.png'
cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars",640,240)
cv2.createTrackbar("Hue Min","TrackBars",15,179,empty)
cv2.createTrackbar("Hue Max","TrackBars",30,179,empty)
cv2.createTrackbar("Sat Min","TrackBars",240,255,empty)
cv2.createTrackbar("Sat Max","TrackBars",255,255,empty)
cv2.createTrackbar("Val Min","TrackBars",30,255,empty)
cv2.createTrackbar("Val Max","TrackBars",160,255,empty)

cap=cv2.VideoCapture(0)
kernel = np.ones((2,2),np.uint8)
while True:
    ret,img = cap.read()
    img= cv2.GaussianBlur(img,(3,3),1)
    imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    h_min = cv2.getTrackbarPos("Hue Min","TrackBars")
    h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
    s_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
    s_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
    v_min = cv2.getTrackbarPos("Val Min", "TrackBars")
    v_max = cv2.getTrackbarPos("Val Max", "TrackBars")
    # print(h_min,h_max,s_min,s_max,v_min,v_max)
    lower = np.array([h_min,s_min,v_min])
    upper = np.array([h_max,s_max,v_max])
    mask = cv2.inRange(imgHSV,lower,upper)
    maskc = cv2.medianBlur(mask, 9)
    imgResult = cv2.bitwise_and(img,img,mask=maskc)
    # imgBlur= cv2.dilate(mask,kernel,iterations=4)
    # imgGray = cv2.cvtColor(imgResult,cv2.COLOR_BGR2GRAY)
    # imgBlur = cv2.bilateralFilter(mask, 5, 40, 10)
    imgBlur = cv2.medianBlur(mask, 5)
    # imgBlur = cv2.GaussianBlur(mask,(21,21),1)
    # imgCanny = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 3, 15)
    imgCanny = cv2.Canny(imgBlur,100,100)
    imgCanny = cv2.GaussianBlur(imgCanny,(3,3),1)
    getContours(imgCanny,img)
    imgStack = stackImages(0.4,([img,imgHSV],[mask,imgResult],[imgCanny,imgBlur]))
    cv2.imshow("Stacked Images", imgStack)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
import cv2

## set width and height of frame
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)


# get background frame
for i in range(10) :
    _, frame = cap.read()
frame = cv2.resize(frame, (640, 480))
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (25,25), 0)
last_frame = gray

while True :
    _, frame = cap.read()
    # handle frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (25,25), 0)

    abs_img = cv2.absdiff(last_frame, gray)

    last_frame = gray
    _, img_mask = cv2.threshold(abs_img, 30,255,cv2.THRESH_BINARY)

    # recognize contours
    contours, _ = cv2.findContours(img_mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours :
        ## filter small contours no need for handle
        if cv2.contourArea(contour) < 900 :
            continue
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 3)

    cv2.imshow("window", frame)

    if cv2.waitKey(1)  == ord('q') :
        break
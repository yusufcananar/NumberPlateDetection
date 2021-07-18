import cv2

#-------CONSTANTS-------------------------------------------------------
minArea = 12000 #pixel^2
cap = cv2.VideoCapture("Resources/nplate.mp4")
#-----------------------------------------------------------------------

#----------TEXT FEATURES---------------------------------------
font = cv2.FONT_HERSHEY_SIMPLEX # font
org = (25, 25)                  # org
fontScale = 0.7                 # fontScale
color = (70, 0, 160)           # Blue color in BGR
thickness = 1                   # Line thickness of 2 px
#--------------------------------------------------------------

#------------CASCADE CLASSIFIER-------------------------------------
#Read Cascade Classifier
nPlateCascade = cv2.CascadeClassifier('Resources/haarcascade_russian_plate_number.xml')
#-------------------------------------------------------------------

#-----------MAIN--------------------------------------
# Read and Show video or webcam feed or Screen Record
lastCoords = []
while True:
    success, img = cap.read()
    if success:
        img = cv2.resize(img, (1280,720))
        img = cv2.GaussianBlur(img, (3, 3), 1)
        imGray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        if len(lastCoords) > 2:
            del lastCoords[0] #To stop overloading the blur

        for rect in lastCoords:
            roi = img[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
            roi = cv2.medianBlur(roi, 11)
            img[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]] = roi

        nPlates = nPlateCascade.detectMultiScale(imGray, 1.1, 2)
        for (x, y, w, h) in nPlates:
            area = w*h
            if area >= minArea:
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
                cv2.putText(img, 'Number Plate', (x,y-10), font, fontScale, color, thickness, cv2.LINE_AA)

                #Blur the Number Plate
                roi = img[y:y+h, x:x+w]
                roi = cv2.medianBlur(roi,11)
                img[y:y + h, x:x + w] = roi
                lastCoords.append([x,y,w,h])


        cv2.imshow("SS", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        print("Video capture is " + success)
        break
#-----------------------------------------------------

#-------CLEAR THE MEMORY------------------------------------------
cv2.destroyAllWindows()
#-----------------------------------------------------------------
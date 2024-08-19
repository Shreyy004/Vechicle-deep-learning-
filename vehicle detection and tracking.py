import cv2
import imutils


cascade_src="cars.xml"
car_cascade=cv2.CascadeClassifier(cascade_src) #loading XML file

cam=cv2.VideoCapture(0)
while True:
    #detected =0
    _,img=cam.read()

    img=imutils.resize(img,width=1000) #resizing the img

    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #converting BGR to greyscale img

    cars=car_cascade.detectMultiScale(gray,1.1,1) #getting coordinates

    for (x,y,w,h) in cars:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2) #drawing a rectangel

    cv2.imshow("Frame",img)

    b=str(len(cars))
    a=int(b)
    n=a
    print("-----------------------------------------------------")

    print("North: %d"%(n))

    if n>=8:
        print("North More Traffic,Please on the Red Signal")

    else:
        print("no traffic")

    if cv2.waitKey(33) ==27:
        break

cam.release()
cv2.destroyAllWindows()

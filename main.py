modelpath = "D:\AI-ComVision\Datasets\yolo.h5"
from imageai import Detection
import cv2

yolo = Detection.ObjectDetection()
yolo.setModelTypeAsYOLOv3()
yolo.setModelPath(modelpath)
yolo.loadModel()
cam = cv2.VideoCapture(0) #0=webcam laptop, 1 = back camera, 2 = usb cam
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, img = cam.read()
    img, preds = yolo.detectCustomObjectsFromImage(input_image=img,custom_objects= None,
                                                   input_type="array",
                                                   output_type="array", minimum_percentage_probability=70,
                                                   display_percentage_probability= False,
                                                   display_object_name= True)
    cv2.imshow("", img)
    if(cv2.waitKey(1) & 0xFF == ord("q") or (cv2.waitKey(1)==27)):
        break

#close camera and and cv2 window
cam.release()
cv2.destroyAllWindows()

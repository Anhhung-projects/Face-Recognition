import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import img_to_array, load_img

path = os.getcwd()
if os.path.isfile('GenderRecognition.h5') == True:
    model = load_model('GenderRecognition.h5')
    print('Model loaded successfully')
else:
    print("Can't load model")
    sys.exit(1)

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
labels = {0:'female', 1:'male'}

def GenderRecognition(img_path,gender=None):
    img = cv2.imread(img_path)
    if img is None:
        print("Error: Can't read image!")
        return None
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
        result = face_detection.process(img_RGB)
        
        if result.detections:
            for detection in result.detections:
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = img.shape
                x, y, w, h = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)

                face = img[y:y+h,x:x+w]
                face = cv2.resize(face,(150,150))
                img_scaled = face/255.0
                reshape = np.reshape(img_scaled,(1,150,150,3))
                img_input = np.vstack([reshape])
                
                result = np.argmax(model.predict(img_input), axis=-1)
                result = result[0]
                if gender not in ['male', 'female']:
                    cv2.rectangle(img,(x-10,y),(x+w,y+h),(0,255,0),4)
                    cv2.rectangle(img,(x-10,y-50),(x+w,y),(255,0,0),-1)
                    
                    if result == 0:
                        cv2.putText(img,labels[0],(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),2)
                    elif result == 1:
                        cv2.putText(img,labels[1],(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),2)
                        
                elif gender == labels[result]:
                    cv2.rectangle(img,(x-10,y),(x+w,y+h),(0,255,0),4)
                    cv2.putText(img,labels[result],(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),2)
                    
                elif gender != labels[result]:
                    cv2.rectangle(img,(x-10,y),(x+w,y+h),(0,0,255),4)
                    cv2.putText(img,labels[result],(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),2)
    
    return img

imgs = os.listdir(os.path.join(path,'Testing'))
output_path = os.path.join(path,'Output')

for img in imgs:
    img_path = os.path.join(path, 'Testing', img)
    label = img.split('_')[0]
    result_img = GenderRecognition(img_path,label)
    if result_img is not None:
        # cv2.imread('Face Detection', result_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()
        print('DONE')
        
        # cv2.imwrite(output_path, result_img)
        # print(f"Saved: {output_path}")
    else:
        print("Error: No image to display!")



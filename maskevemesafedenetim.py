import cv2
import numpy as np
import math
from threading import Thread

SOCIAL_DISTANCE_THRESH = 200
inputvalue = set()
outputvalue = set()

def inputpage(): 

  cap = cv2.VideoCapture("C:/Users/Dilek/Desktop/maske ve sosyal mesafe kendi eğtiim/123.mp4")
  #cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
  #cap = cv2.imread("C:/Users/Dilek/Desktop/maske ve sosyal mesafe kendi eğtiim/inputphoto.jpg")
  while(True):
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (660, 720))

    frame_width = frame.shape[1]
    frame_height = frame.shape[0]

    frame_blob = cv2.dnn.blobFromImage(
        frame, 1 / 255, (416, 416), swapRB=True, crop=False)

    labels = ["maske_var", "maske_yok", "insan"]

    colors = ["255,0,0", "0,0,255", "178,34,34"]
    colors = [np.array(color.split(",")).astype("int") for color in colors]
    colors = np.array(colors)
    colors = np.tile(colors, (18, 1))

    model = cv2.dnn.readNetFromDarknet("yolov4.cfg", "yolov4_last.weights")

    layers = model.getLayerNames()
    output_layer = [layers[layer[0] - 1]
                    for layer in model.getUnconnectedOutLayers()]

    model.setInput(frame_blob)
    detection_layers = model.forward(output_layer)

    ids_list = []
    boxes_list = []
    confidences_list = []

    for detection_layer in detection_layers:
        for object_detection in detection_layer:

            scores = object_detection[5:]
            predicted_id = np.argmax(scores)
            confidence = scores[predicted_id]

            if confidence > 0.20:

                label = labels[predicted_id]
                bounding_box = object_detection[0:4] * np.array(
                    [frame_width, frame_height, frame_width, frame_height])
                (box_center_x, box_center_y, box_width,
                 box_height) = bounding_box.astype("int")

                start_x = int(box_center_x - (box_width / 2))
                start_y = int(box_center_y - (box_height / 2))

                ids_list.append(predicted_id)
                confidences_list.append(float(confidence))
                boxes_list.append(
                    [start_x, start_y, int(box_width), int(box_height)])

    max_ids = cv2.dnn.NMSBoxes(boxes_list, confidences_list, 0.5, 0.4)

    body_coord_lst = []
    confidence_lst = []
    bounding_box_lst = []

    for max_id in max_ids:

        max_class_id = max_id[0]
        box = boxes_list[max_class_id]

        start_x = box[0]
        start_y = box[1]
        box_width = box[2]
        box_height = box[3]

        predicted_id = ids_list[max_class_id]
        label = labels[predicted_id]
        confidence = confidences_list[max_class_id]

        end_x = start_x + box_width
        end_y = start_y + box_height

        box_color = colors[predicted_id]
        box_color = [int(each) for each in box_color]
        

        # sosyal mesafe
        if label == labels[2]:
            
            body_coord_lst.append((start_x + box_width / 2, start_y + box_height / 2))
            confidence_lst.append(confidence)
            bounding_box_lst.append((start_x, box_width, start_y, box_height))

        label_str = "{}:{:.2f}%".format(label, confidence * 100)
       
        
        if label != labels[2]:
            
            cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), box_color, 2)
            cv2.rectangle(frame, (start_x - 1, start_y),
                          (end_x + 1, start_y - 30), box_color, -1)
            cv2.putText(frame, label_str, (start_x, start_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
    
    body_count = len(body_coord_lst)
    for i in range(body_count):
        #print("giriş ekranı i için",i)
        is_close = False
        for j in range(body_count):
            #print ("giriş ekranı j için",j)
            inputvalue.add(i)
            
            if i == j:
                continue
            distance = math.sqrt((body_coord_lst[i][0] - body_coord_lst[j][0]) ** 2 + (
                body_coord_lst[i][1] - body_coord_lst[j][1]) ** 2)
            if distance < SOCIAL_DISTANCE_THRESH:
                is_close = True
                break

        start_x, box_width, start_y, box_height = bounding_box_lst[i]
        end_x = start_x + box_width
        end_y = start_y + box_height
        confidence = confidence_lst[i]

        box_color = (0, 0, 255) if is_close else (255, 0, 0)

        label = "{}:{:.2f}%".format('insan', confidence * 100)
        cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), box_color, 2)
        cv2.rectangle(frame, (start_x - 1, start_y), (end_x + 1, start_y - 30), box_color, -1)
        cv2.putText(frame, label, (start_x, start_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        text= "Giren insan sayisi: {}".format(len(inputvalue))
        cv2.putText(frame, text, (10, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 3)
        
        result = len (inputvalue) - len(outputvalue)
        text = "Kalan insan sayisi: {}".format(result)
        cv2.putText(frame, text, (10, frame.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 3)
    if ret == True:
      cv2.imshow('Giris Denetimi',frame)
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    else :
        break
  cap.release()
  cv2.destroyAllWindows()
    
  
def outputpage(): 
  capout = cv2.VideoCapture("C:/Users/Dilek/Desktop/maske ve sosyal mesafe kendi eğtiim/11.mp4")
  #capout = cv2.VideoCapture(1)
  while(True):
    retout, frameout = capout.read()

    frameout = cv2.flip(frameout, 1)
    frameout = cv2.resize(frameout, (660, 720))

    frame_width = frameout.shape[1]
    frame_height = frameout.shape[0]

    frame_blob = cv2.dnn.blobFromImage(
        frameout, 1 / 255, (416, 416), swapRB=True, crop=False)

    labels = ["maske_var", "maske_yok", "insan"]

    colors = ["255,0,0", "0,0,255", "178,34,34"]
    colors = [np.array(color.split(",")).astype("int") for color in colors]
    colors = np.array(colors)
    colors = np.tile(colors, (18, 1))

    model = cv2.dnn.readNetFromDarknet("yolov4.cfg", "yolov4_last.weights")

    layers = model.getLayerNames()
    output_layer = [layers[layer[0] - 1]
                    for layer in model.getUnconnectedOutLayers()]

    model.setInput(frame_blob)
    detection_layers = model.forward(output_layer)

    ids_list = []
    boxes_list = []
    confidences_list = []

    for detection_layer in detection_layers:
        for object_detection in detection_layer:

            scores = object_detection[5:]
            predicted_id = np.argmax(scores)
            confidence = scores[predicted_id]

            if confidence > 0.20:

                label = labels[predicted_id]
                bounding_box = object_detection[0:4] * np.array(
                    [frame_width, frame_height, frame_width, frame_height])
                (box_center_x, box_center_y, box_width,
                 box_height) = bounding_box.astype("int")

                start_x = int(box_center_x - (box_width / 2))
                start_y = int(box_center_y - (box_height / 2))

                ids_list.append(predicted_id)
                confidences_list.append(float(confidence))
                boxes_list.append(
                    [start_x, start_y, int(box_width), int(box_height)])

    max_ids = cv2.dnn.NMSBoxes(boxes_list, confidences_list, 0.5, 0.4)

    body_coord_lst = []
    confidence_lst = []
    bounding_box_lst = []

    for max_id in max_ids:

        max_class_id = max_id[0]
        box = boxes_list[max_class_id]

        start_x = box[0]
        start_y = box[1]
        box_width = box[2]
        box_height = box[3]

        predicted_id = ids_list[max_class_id]
        label = labels[predicted_id]
        confidence = confidences_list[max_class_id]

        end_x = start_x + box_width
        end_y = start_y + box_height

        box_color = colors[predicted_id]
        box_color = [int(each) for each in box_color]
        

        # sosyal mesafe
        if label == labels[2]:
            
            body_coord_lst.append((start_x + box_width / 2, start_y + box_height / 2))
            confidence_lst.append(confidence)
            bounding_box_lst.append((start_x, box_width, start_y, box_height))

        label_str = "{}:{:.2f}%".format(label, confidence * 100)
       
        
        if label != labels[2]:
            
            cv2.rectangle(frameout, (start_x, start_y), (end_x, end_y), box_color, 2)
            cv2.rectangle(frameout, (start_x - 1, start_y),
                          (end_x + 1, start_y - 30), box_color, -1)
            cv2.putText(frameout, label_str, (start_x, start_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
    
    body_count = len(body_coord_lst)
    for i in range(body_count):
        is_close = False
        for j in range(body_count):
            outputvalue.add(i)
            
            if i == j:
                continue
            distance = math.sqrt((body_coord_lst[i][0] - body_coord_lst[j][0]) ** 2 + (
                body_coord_lst[i][1] - body_coord_lst[j][1]) ** 2)
            if distance < SOCIAL_DISTANCE_THRESH:
                is_close = True
                break

        start_x, box_width, start_y, box_height = bounding_box_lst[i]
        end_x = start_x + box_width
        end_y = start_y + box_height
        confidence = confidence_lst[i]

        box_color = (0, 0, 255) if is_close else (255, 0, 0)

        label = "{}:{:.2f}%".format('insan', confidence * 100)
        cv2.rectangle(frameout, (start_x, start_y), (end_x, end_y), box_color, 2)
        cv2.rectangle(frameout, (start_x - 1, start_y), (end_x + 1, start_y - 30), box_color, -1)
        cv2.putText(frameout, label, (start_x, start_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        text= "Cikan insan sayisi: {}".format(len(outputvalue))
        cv2.putText(frameout, text, (10, frameout.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 3)
    
    if retout == True:
     cv2.imshow('Cikis Denetimi',frameout)
     if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    else :
        break     
  capout.release()
  cv2.destroyAllWindows()

if __name__ == '__main__':
    inp = Thread(target = inputpage)
    out = Thread(target = outputpage)

    inp.start()
    out.start()


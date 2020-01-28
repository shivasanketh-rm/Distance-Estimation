from __future__ import print_function
import torch
from torch.autograd import Variable
import cv2
import time
from imutils.video import FPS
import argparse
import csv
import math
import multiprocessing
from time import sleep

from distance_estimator import distance_estimator


parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--weights', default='./ssd300_mAP_77.43_v2.pth',
                    type=str, help='Trained state_dict file path')
parser.add_argument('--type', default= 'camera',
                    type=str, help='Input type - camera, or file path')
parser.add_argument('--file_path', default= './video.avi',
                    type=str, help='file path')
parser.add_argument('--cuda', default=False, type=bool,
                    help='Use cuda')
args = parser.parse_args()

COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
FONT = cv2.FONT_HERSHEY_SIMPLEX

def cv2_demo(net, transform, frame, de, dist_queue, output_frame_queue):
    def predict(frame, de):
        height, width = frame.shape[:2]
        x = torch.from_numpy(transform(frame)[0]).permute(2, 0, 1)
        x = Variable(x.unsqueeze(0))
        y = net(x)  # forward pass
        detections = y.data
        
        dist = 0

        # scale each detection back up to the image
        scale = torch.Tensor([width, height, width, height])
        for i in range(detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= 0.6:
                pt = (detections[0, i, j, 1:] * scale).cpu().numpy()

                x,y = pt[0], pt[1]
                h,w = pt[3]-pt[1],pt[2]-pt[0]

                if labelmap[i - 1] == "person":
                    dist = de.estimator(h,w)
                    #print("dist = ", dist)
                    cv2.rectangle(frame,
                                (int(pt[0]), int(pt[1])),
                                (int(pt[2]), int(pt[3])),
                                COLORS[i % 3], 2)
                    cv2.putText(frame, labelmap[i - 1] + str(round(dist,2)), (int(pt[0]), int(pt[1])),
                                FONT, 1.5, COLORS[i % 3], 2, cv2.LINE_AA)
                    #print("person = ", labelmap[i - 1])
                    #print("person type = ", type(labelmap[i - 1]))
                    
                j += 1
        return frame, dist

    frame, dist = predict(frame, de)
    output_frame_queue.put(frame)
    dist_queue.put(dist)

    

def input_output(input_type, input_file, input_frame_queue, dist_queue, output_frame_queue, p2):
    print("camera")
    dist = 0
    
    def image_inference(file_path):
        frame = cv2.imread(file_path)
        #cv2.imshow('Frame',frame)
        dist = 0
        input_frame_queue.put(frame)
        #Wait for the Inference to complete
        sleep(3)
        if dist_queue.empty() == False:
            dist = dist_queue.get()
        cv2.putText(frame, str(round(dist,2)), (35,35),
                            FONT, 1.5,(255, 0, 0) , 2, cv2.LINE_AA)
        cv2.imshow('Frame',frame)
        cv2.waitKey(0)
        return 0

    while True:
        #Check if input is camera or image or video
        if input_type == 'camera':
            video_file = 0
        else:
            video_file = input_file
            if input_type == 'image':
                image_inference(args.file_path)
                exit(0)

        cap = cv2.VideoCapture(video_file)
        # Check if camera opened successfully
        if (cap.isOpened()== False): 
            print("Error opening video stream or file")

        #Frame Count
        count = 0
        while(cap.isOpened()):
        # Capture frame-by-frame
            ret, frame = cap.read()
            if ret == True:
                print("Frame count is {}" .format(count))
                # Display the resulting frame
                key = cv2.waitKey(1) & 0xFF
                if count % 10 == 0:
                    input_frame_queue.put(frame)
                    #If Distance queue is not empty get distance (Updated from inference at function - cv2_demo->predict) 
                    if dist_queue.empty() == False:
                        dist = dist_queue.get()
                    #If Output frame queue is not empty get output frame with bounding box (Updated from inference at function - cv2_demo->predict) 
                    if output_frame_queue.empty() == False:
                        frame = output_frame_queue.get()
                count += 1
                #Display Distance onto Output Frame
                cv2.putText(frame, str(round(dist,2)), (35,35),
                                FONT, 1.5,(255, 0, 0) , 2, cv2.LINE_AA)
                cv2.imshow('Frame',frame)
                #cv2.waitKey()
                if key == ord('q'):  # quit
                    exit()
            else: 
                break
    return 0


def inference(net, transform, input_frame_queue, dist_queue, output_frame_queue):
    #Create instance of Distance Estimator class
    de = distance_estimator()
    while True:
        print("inference")
        #If Input frame queue is not empty get frame 
        if input_frame_queue.empty() == False:
            frame = input_frame_queue.get()
            cv2_demo(net.eval(), transform, frame, de, dist_queue, output_frame_queue)
    return 0


if __name__ == '__main__':
    import sys
    from os import path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

    from data import BaseTransform, VOC_CLASSES as labelmap
    from ssd import build_ssd

    net = build_ssd('test', 300, 21)    # initialize SSD
    net.load_state_dict(torch.load(args.weights, map_location="cpu"))
    transform = BaseTransform(net.size, (104/256.0, 117/256.0, 123/256.0))

    input_frame_queue = multiprocessing.Queue(maxsize=1)
    output_frame_queue = multiprocessing.Queue(maxsize=1)
    dist_queue = multiprocessing.Queue(maxsize=1)

    #Intitiate both Processes
    #P1 is for reading Input and displaying Output
    p1 = multiprocessing.Process(target=input_output, args=(args.type,args.file_path, input_frame_queue,dist_queue,
                                    output_frame_queue, ))
    #P2 is for SSD and Distance Estimation
    p2 = multiprocessing.Process(target=inference, args=(net,transform, input_frame_queue,dist_queue,
                                    output_frame_queue ))
     
     
  
    # starting process 1 
    p1.start() 
    # starting process 2 
    p2.start() 
  
    # wait until process 1 is finished 
    p1.join() 
    # wait until process 2 is finished 
    p2.join() 

    # cleanup
    cv2.destroyAllWindows()

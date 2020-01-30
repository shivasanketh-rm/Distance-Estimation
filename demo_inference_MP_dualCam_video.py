'''
Input Format -
python3 demo_inference_MP_3p.py --type 'image' --file_path 'file1.jpg' --file_path2 'file2.jpg'

'''

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
parser.add_argument('--file_path2', default= './video.avi',
                    type=str, help='file path')
parser.add_argument('--cuda', default=False, type=bool,
                    help='Use cuda')
args = parser.parse_args()

COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
FONT = cv2.FONT_HERSHEY_SIMPLEX

def cv2_demo(net, transform, frame, de, distance_queue, out_frame_queue):
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

    #clear values and then put new values
    if not out_frame_queue.empty():
        out_frame_queue.get()
    out_frame_queue.put(frame)
    if not distance_queue.empty():
        distance_queue.get()
    distance_queue.put(dist)

    

def input_output(input_type, input_file, input_file2, input_frame_queue, input_frame_queue2, dist_queue,dist_queue2, output_frame_queue, output_frame_queue2):
    print("camera")
    dist = dist2 = 0
    
    #Function for image as input 
    def image_inference(file_path, file_path2):
        frame = cv2.imread(file_path)
        frame2 = cv2.imread(file_path2)
        #cv2.imshow('Frame',frame)
        dist = dist2 = 0
        input_frame_queue.put(frame)
        input_frame_queue2.put(frame2)
        #Wait for the Inference to complete
        sleep(6)
        if dist_queue.empty() == False and dist_queue2.empty() == False:
            dist = dist_queue.get()
            dist2 = dist_queue2.get()
            print("dist = ", dist)
            print("dist2 = ", dist2)
        cv2.putText(frame, 'Front Cam: ' + str(round(dist,2)), (35,35),
                            FONT, 1,(255, 0, 0) , 2, cv2.LINE_AA)
        cv2.putText(frame, 'Side Cam: ' + str(round(dist2,2)), (35,80),
                            FONT, 1,(255, 0, 0) , 2, cv2.LINE_AA)
        cv2.imshow('Front Cam',frame)
        cv2.imshow('Side Cam',frame2)
        cv2.waitKey(0)
        return 0

    while True:
        #Check if input is camera or image or video
        if input_type == 'camera':
            video_file = 0
        else:
            video_file = input_file
            video_file2 = input_file2
            if input_type == 'image':
                image_inference(input_file, input_file2)
                exit(0)

        #video or camera
        cap = cv2.VideoCapture(video_file)
        cap2 = cv2.VideoCapture(video_file2)
        # Check if camera/video opened successfully
        if (cap.isOpened()== False and cap2.isOpened() == False): 
            print("Error opening video stream or file")

        #Frame Count
        count = 0
        while(cap.isOpened() and cap2.isOpened()):
        # Capture frame-by-frame
            
            ret, frame = cap.read()
            ret2, frame2 = cap2.read()
            if ret == True and ret2 == True:
                print("Frame count is {}" .format(count))
                # Display the resulting frame
                key = cv2.waitKey(1) & 0xFF
                if count % 10 == 0:

                    #clear values and then put new values
                    if (input_frame_queue.empty() == False):
                        input_frame_queue.get()
                    input_frame_queue.put(frame)
                    if (input_frame_queue2.empty() == False):
                        input_frame_queue2.get()
                    input_frame_queue2.put(frame2)

                    #If Distance queue is not empty get distance (Updated from inference at function - cv2_demo->predict) 
                    if dist_queue.empty() == False and dist_queue2.empty() == False:
                        dist = dist_queue.get()
                        dist2 = dist_queue2.get()
                        print("dist = ", dist)
                        print("dist2 = ", dist2)
                    #If Output frame queue is not empty get output frame with bounding box (Updated from inference at function - cv2_demo->predict) 
                    if output_frame_queue.empty() == False:
                        frame = output_frame_queue.get()
                count += 1
                #Display Distance onto Output Frame
                cv2.putText(frame, 'Front Cam: ' + str(round(dist,2)), (35,35),
                                FONT, 1,(255, 0, 0) , 2, cv2.LINE_AA)
                cv2.putText(frame, 'Side Cam: ' + str(round(dist2,2)), (35,80),
                                FONT, 1,(255, 0, 0) , 2, cv2.LINE_AA)
                cv2.imshow('Front Cam',frame)
                cv2.imshow('Side Cam',frame2)
                sleep(0.5)
                #cv2.waitKey()
                if key == ord('q'):  # quit
                    exit()
            else: 
                break
    return 0


def inference(net, transform, input_frame_queue, dist_queue, output_frame_queue):
    #Create instance of Distance Estimator class
    de1 = distance_estimator()
    while True:
        #print("inference")
        #If Input frame queue is not empty get frame 
        if input_frame_queue.empty() == False:
            frame = input_frame_queue.get()
            cv2_demo(net.eval(), transform, frame, de1, dist_queue, output_frame_queue)
    return 0


def inference2(net, transform, input_frame_queue2, dist_queue2, output_frame_queue2):
    #Create instance of Distance Estimator class
    de2 = distance_estimator()
    while True:
        #print("inference2")
        #If Input frame queue is not empty get frame 
        if input_frame_queue2.empty() == False:
            frame2 = input_frame_queue2.get()
            cv2_demo(net.eval(), transform, frame2, de2, dist_queue2, output_frame_queue2)
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
    input_frame_queue2 = multiprocessing.Queue(maxsize=1)
    output_frame_queue2 = multiprocessing.Queue(maxsize=1)
    dist_queue2 = multiprocessing.Queue(maxsize=1)

    #Intitiate both Processes
    #P1 is for reading Input and displaying Output
    p1 = multiprocessing.Process(target=input_output, args=(args.type,args.file_path, args.file_path2, input_frame_queue,
                                    input_frame_queue2, dist_queue,dist_queue2, output_frame_queue, output_frame_queue2 ))
    #P2 is for SSD and Distance Estimation
    p2 = multiprocessing.Process(target=inference, args=(net,transform, input_frame_queue,dist_queue,
                                    output_frame_queue, ))

    p3 = multiprocessing.Process(target=inference2, args=(net,transform, input_frame_queue2,dist_queue2,
                                    output_frame_queue2, )) 
     
  
    # starting processes 
    p1.start() 
    p2.start() 
    p3.start() 
  
    # wait until process 1 is finished 
    p1.join() 
    # wait until process 2 is finished 
    p2.join() 
    # wait until process 3 is finished 
    p3.join() 

    # cleanup
    cv2.destroyAllWindows()

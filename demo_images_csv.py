from __future__ import print_function
import torch
from torch.autograd import Variable
import cv2
import time
from imutils.video import FPS
import argparse
import csv



parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--weights', default='./ssd300_mAP_77.43_v2.pth',
                    type=str, help='Trained state_dict file path')
#parser.add_argument('--weights', default='weights/ssd_300_VOC0712.pth',
#                    type=str, help='Trained state_dict file path')
parser.add_argument('--video_path',
                    type=str, help='Input Video file path')
parser.add_argument('--csv_file_path',
                    type=str, help='CSV file path')
parser.add_argument('--class_info',
                    type=str, help='Class info')
parser.add_argument('--cuda', default=False, type=bool,
                    help='Use cuda')
args = parser.parse_args()

COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
FONT = cv2.FONT_HERSHEY_SIMPLEX
 
class writeToCsv():
    def __init__(self):
        #print("inside CSV init")
        with open(args.csv_file_path, mode='w') as employee_file:
            #print("inside CSV init loop")
            employee_writer = csv.writer(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

            employee_writer.writerow(["x","y","h","w","c_x","c_y","class"])

    def writer(self, array):
        #print("inside CSV init")
        with open(args.csv_file_path, mode='a') as employee_file:
            employee_writer = csv.writer(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

            employee_writer.writerow(array)
    #employee_writer.writerow(['Erica Meyers', 'IT', 'March'])

def cv2_demo(net, transform):
    def predict(frame):
        height, width = frame.shape[:2]
        x = torch.from_numpy(transform(frame)[0]).permute(2, 0, 1)
        x = Variable(x.unsqueeze(0))
        y = net(x)  # forward pass
        detections = y.data
        
        # scale each detection back up to the image
        scale = torch.Tensor([width, height, width, height])
        for i in range(detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= 0.6:
                pt = (detections[0, i, j, 1:] * scale).cpu().numpy()

                x,y = pt[0], pt[1]
                h,w = pt[3]-pt[1],pt[2]-pt[0]
                print("x = {},y = {},h = {},w = {}".format(x,y,h,w) )
                print("x = {},y = {},h = {},w = {}".format(int(x),int(y),int(h),int(w)) )
                center = (int(x + w / 2), int(y + h / 2))
                
                csv_writer.writer([x,y,h,w,center[0],center[1],args.class_info])
                axis_major = w / 2
                axis_minor = h / 2
                '''cv2.ellipse(frame,
                    center=center,
                    axes=(int(axis_major), int(axis_minor)),
                    angle=0,
                    startAngle=0,
                    endAngle=360,
                    color=COLORS[i % 3],
                    thickness=2)'''

                '''cv2.rectangle(frame,
                              (int(pt[0]), int(pt[1])),
                              (int(pt[2]), int(pt[3])),
                              COLORS[i % 3], 2)
                cv2.putText(frame, labelmap[i - 1], (int(pt[0]), int(pt[1])),
                            FONT, 1.5, COLORS[i % 3], 2, cv2.LINE_AA)'''
                j += 1
        return frame

    #read the frame
    #frame = cv2.imread(args.input_frame)
    #print("frame = ", frame)
    #print("shape of frame = ", frame.shape)
    #print("type of frame = ", type(frame))
    cap = cv2.VideoCapture(args.video_path)
    
    # Check if camera opened successfully
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
    
    # Read until video is completed
    csv_writer = writeToCsv()
    num = 0
    while(cap.isOpened()):
    # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            num += 1
            print("Frame count is {}" .format(num))
            # Display the resulting frame
            #cv2.imshow('Frame',frame)
            frame = predict(frame)

        else: 
            break


if __name__ == '__main__':
    import sys
    from os import path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

    from data import BaseTransform, VOC_CLASSES as labelmap
    from ssd import build_ssd

    net = build_ssd('test', 300, 21)    # initialize SSD
    net.load_state_dict(torch.load(args.weights, map_location="cpu"))
    transform = BaseTransform(net.size, (104/256.0, 117/256.0, 123/256.0))


    cv2_demo(net.eval(), transform)
    # cleanup
    cv2.destroyAllWindows()

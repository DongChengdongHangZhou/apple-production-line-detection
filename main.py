## -*- coding: utf-8 -*-

import numpy as np
import cv2
import torch
import numpy as np
from resnet_model import resnext50_32x4d
import cv2


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
resnet = resnext50_32x4d(pretrained=None, num_classes = 2).to(device)
resnet.eval()
checkpoint = torch.load('./checkpoints/inceptionNet0001.pth')
resnet.load_state_dict(checkpoint['inceptionNet'])

def get_feature(image):
    image = np.array([image[:, :, 2], image[:, :, 1], image[:, :, 0]])
    image = torch.tensor(image)
    image = image.cuda()
    image = (image-127.5)/128
    output = resnet(image.unsqueeze(0))
    return output.detach().cpu()

def calc_width_height(count):
    frame_resize = cv2.resize(frame,(160,90))
    frame_resize = cv2.medianBlur(frame_resize,7)
    mask = np.ones((90,160))
    frame_resize_r,frame_resize_g,frame_resize_b = frame_resize[:,:,0].copy(),frame_resize[:,:,1].copy(),frame_resize[:,:,2].copy()
    mask[frame_resize_r <=110] = 0
    mask[frame_resize_r >=155] = 0
    mask[frame_resize_g <=110] = 0
    width = np.sum(mask,axis=1).max()*12
    height = np.sum(mask,axis=0).max()*12
    dict = {1:(764,820),2:(826,812),3:(822,872),4:(830,808),5:(870,886),6:(864,838)}
    return dict[count]

def state(frame):
    frame_resize = cv2.resize(frame,(80,45))
    mask = np.ones((45,80))
    count = 0
    x = 0
    y = 0
    frame_resize_r,frame_resize_g,frame_resize_b = frame_resize[:,:,0].copy(),frame_resize[:,:,1].copy(),frame_resize[:,:,2].copy()
    mask[frame_resize_r <=110] = 0
    mask[frame_resize_r >=155] = 0
    mask[frame_resize_g <=110] = 0
    mask[frame_resize_g >=190] = 0
    mask[frame_resize_b <=195] = 0
    count = mask.sum()
    apple_location_x,apple_location_y = np.where(mask != 0)[0].sum(),np.where(mask != 0)[1].sum()
    y = apple_location_y/(count+1)
    x = apple_location_x/(count+1)
    ret = []
    ret.append(x)
    ret.append(y)
    if count>250 and y>50 and y<51:
        ret.append(1)
    else:
        ret.append(0)
    return ret

if __name__ == '__main__':
    SAVE_VIDEO = False
    CNT = 0
    flag = 0
    state_list = []
    cap = cv2.VideoCapture('apple.mp4')
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    if SAVE_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('screen.avi', fourcc, 20.0, (1920, 1080))
    count_apple = 0
    while cap.isOpened():
        CNT = CNT + 1
        ret, frame = cap.read()
        try:
            frame[12:55,80:1857] = 240
        except:
            pass
        if not ret:
            print("Ending")
            break
        current_state = state(frame)
        state_list.append(current_state[2])
        if current_state[2] == 1 and last_state[2] == 0 and count_apple!=6:
            count_apple += 1
        if current_state[2] == 1:
            crop_apple = frame
            img_grey_level = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(img_grey_level, (35, 35), 30,30)
            ret, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU)
            mask[0:50, :] = 0
            mask = cv2.dilate(mask, kernel, iterations=5)
            #image, contours, hierarchy = cv2.findContours(mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
            mask = cv2.multiply(mask, 1 / 255)
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            x_ = int(24 * current_state[0])
            y_ = int(24 * current_state[1])
            cv2.line(frame,(y_-236,x_-216),(y_+276,x_-216),(0,255,0),10)
            cv2.line(frame,(y_-236,x_-216),(y_-236,x_+296),(0,255,0),10)
            cv2.line(frame,(y_+276,x_-216),(y_+276,x_+296),(0,255,0),10)
            cv2.line(frame,(y_-236,x_+296),(y_+276,x_+296),(0,255,0),10)
            if current_state[1]>40.5:
                width,height = calc_width_height(count_apple)
                crop_pattern = frame[x_ - 216:x_ + 296, y_ - 236:y_ + 276]
                if count_apple == 1 or count_apple == 2 or count_apple == 3:
                    stem = 'no stem'
                if count_apple == 4 or count_apple == 5 or count_apple == 6:
                    stem = 'has stem'
        last_state = current_state
        try:
            frame[80:336,1600:1856] = cv2.resize(crop_pattern,(256,256))
            if count_apple==3:
                cv2.circle(frame,(1817,169),21,(0,255,0),6)
                black_hole = 'Yes'
            if count_apple >0 and count_apple!=3:
                black_hole = 'No'
            text = "count:"+str(count_apple)+"   width:"+str(width)+"   height:"+str(height)+"   stem:"+stem+"   black hole:"+black_hole
        except:
            text = "count:"+str(count_apple)+" width:"+'         '+"height:"+'         '+"stem:"+'        '+'black hole:'+'       '
        cv2.putText(frame, text, (150, 46), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.8, (30, 45, 155), 2)
        cv2.namedWindow("frame", 0)
        cv2.resizeWindow("frame", 960, 540)
        cv2.imshow('frame', frame)
        if SAVE_VIDEO:
            out.write(frame)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    if SAVE_VIDEO:
        out.release()
    cv2.destroyAllWindows()

# created for the purpose of being efficient
import numpy as np
import cv2
import sys
import time


def init_video(filepath):
    return cv2.VideoCapture(filepath)


def resize_frame(frame):
    return cv2.resize(frame,(0,0),fx=0.25,fy=0.25)


def parse_frames(filepath):
    vid = cv2.VideoCapture(filepath)
    size = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = [cv2.resize(vid.read()[1],(0,0),fx=0.25,fy=0.25) for x in range(size)]
    return frames



def canny_edge_detector(frame):
    pass




def main_loop(filepath):
    frames = parse_frames(filepath)
    for frame in frames:
        edge_frame = canny_edge_detector(frame)
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    vid.release()
    cv2.destroyAllWindows()



'''
def main_loop(filepath):
    vid = init_video(filepath)
    while vid.isOpened():
        ret,frame = vid.read()
        #frames = [vid.grap() for x in range(10) if vid.isOpened()]
        #for f in frames:
        #    cv2.imshow('frame',f)
        small_frame = cv2.resize(frame,(0,0),fx=0.25,fy=0.25)

        # do stuff to video here

        cv2.imshow('frame',small_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    vid.release()
    cv2.destroyAllWindows()
'''



def driver():
    filepath = 'vid/aerial.mp4'
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    main_loop(filepath)


if __name__ == '__main__':
    driver()



# canny made to be efficient

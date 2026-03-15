#!/usr/bin/env python3

# ROS node that grabs frames from the Kinova camera and runs the card detector on them

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from card_detector import ShapeMatcher

bridge = CvBridge()
matcher = ShapeMatcher()

def callback(msg):
    # convert ROS image to OpenCV format
    frame = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
    
    # run the detector
    annotated, detections = matcher.process_frame(frame)

    # print whatever was found
    for d in detections:
        print(f"Detected: {d.card_name}  mse={d.mse}")

# subscribe to the Kinova wrist camera topic
rospy.init_node("card_detector_node")
rospy.Subscriber("/right/camera/color/image_raw", Image, callback)
rospy.spin()

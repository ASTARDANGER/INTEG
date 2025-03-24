#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64MultiArray
from franka_msgs.msg import FrankaState
from cv_bridge import CvBridge
import cv2
import time
import os
from pynput import keyboard

class IntegVisionNode:
    def __init__(self):
        # Initialize the node
        rospy.init_node('integ_vision_node', anonymous=True)

        # Subscribers 
        ###TODO###
        rospy.Subscriber("/franka_state_controller/franka_states", FrankaState, self.cart_states_callback)
        rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback)
        ###TODO###
        
        # Init variables
        self.bridge = CvBridge()
        self.latest_image = None
        self.current_cartesian_pose = None
        self.recording = False

        # Define the folder to save images
        self.save_folder = "/home/ecn/Pictures/integ_saved_images"
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

        # Define the file to save Cartesian poses
        self.cart_poses_file = os.path.join(self.save_folder, "cart_poses.txt")
        with open(self.cart_poses_file, 'w') as f:
            f.write("Image Name, TCP poses (t00, t10, t20, t30, t10, t11, t12, t13, t20, t21, t22, t23, t30, t31, t32, t33) \n")
            
        # Start keyboard listener
        self.listener = keyboard.Listener(on_press=self.on_key_press)
        self.listener.start()

    def actions_callback(self, msg):
        # Activate fonctions (e.g. launch self.save_image_and_robot_state() to record a single image+pose)
        ###TODO###
        self.save_image_and_robot_state(self)
        ###TODO###
    
    def cart_states_callback(self, msg):
        # Store the current Cartesian pose of the robot
        ###TODO###
        self.current_cartesian_pose = msg.O_T_EE
        ###TODO###

    def image_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image and store it
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            if self.recording:
                self.save_image_and_robot_state()
        except Exception as e:
            rospy.logerr(f"Failed to process image: {e}")

    def save_image_and_robot_state(self):
        if self.latest_image is None:
            rospy.logwarn("No image received yet. Camera might not be active.")
            return

        if self.current_cartesian_pose is None:
            rospy.logwarn("No Cartesian pose received yet. Robot might not be active.")
            return

        try:
            # Save the image & Save the Cartesian pose to the file
            ###TODO###
            # Generate unique image filename
            image_filename = f"image_{rospy.Time.now().to_sec()}.jpg"
            image_path = os.path.join(self.save_folder, image_filename)

            # Save the image
            cv2.imwrite(image_path, self.latest_image)

            if not self.recording:
                # Save Cartesian pose to the file
                with open(self.cart_poses_file, 'a') as f:
                    rospy.loginfo("Screenshot saved")
                    pose_str = ", ".join([str(i) for i in self.current_cartesian_pose])
                    f.write(f"{image_filename}, {pose_str}\n")

            rospy.loginfo(f"Saved image and pose: {image_filename}")
            ###TODO###
        except Exception as e:
            rospy.logerr(f"Failed to save image or robot state: {e}")

    def on_key_press(self, key):
        try:
            if key.char == 'p':
                self.save_image_and_robot_state()
            if key.char == 'v':
                self.recording = not self.recording
        except AttributeError:
            pass
            
    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        node = IntegVisionNode()
        node.run()
    except rospy.ROSInterruptException:
        pass

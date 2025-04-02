# ROS
import rospy
from sensor_msgs.msg import Image, CameraInfo
from stereo_msgs.msg import DisparityImage
import cv_bridge
import message_filters
from tf2_ros import TransformListener, Buffer, TransformException
import tf2_geometry_msgs
from std_srvs.srv import Trigger, TriggerResponse
from visualization_msgs.msg import MarkerArray, Marker

import PIL.Image as PILImage
import math
import numpy as np

# Our module
import digit_recognizer as recognizer

BOX_SIZE = 1


class OnlineDigitCounter:
    def __init__(self):
        self.camera_info = None
        self.depth_image = None
        self.state = None
        self.enable_signal = False
        self.digits_and_positions = []
        self.marker_array = MarkerArray()
        self.cv_bridge = cv_bridge.CvBridge()
        self.tfBuffer = Buffer()
        self.listener = TransformListener(self.tfBuffer)
        self.enable_signal_server = rospy.Service("enable_perception_signal", Trigger, self.enable_signal_callback)
        self.left_camera_info_sub = rospy.Subscriber("/front/left/camera_info", CameraInfo, self.camera_info_callback)
        self.left_image_sub = message_filters.Subscriber("/front/left/image_raw", Image)
        self.disparity_sub = message_filters.Subscriber("/front/disparity", DisparityImage)
        self.sync = message_filters.TimeSynchronizer([self.left_image_sub, self.disparity_sub], queue_size=1)
        self.marker_pub = rospy.Publisher("/digits", MarkerArray, queue_size=1)
        self.sync.registerCallback(self.stereo_camera_callback)
        print("OnlineNumberCounter initialized")

    def enable_signal_callback(self, req):
        self.enable_signal = True
        return TriggerResponse(success=True)

    def visualize_marker(self, x, y, z, target_frame):
        marker = Marker()
        marker.header.frame_id = target_frame
        marker.header.stamp = rospy.Time.now()
        marker.ns = "basic_shapes"
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = z
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1

        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        marker.lifetime = rospy.Duration(0)
        self.marker_array.markers.append(marker)

    def merge_digit(self, digit, position, target_frame="map"):
        # transform position in camera frame to target frame
        pose = tf2_geometry_msgs.PoseStamped()
        pose.header.frame_id = "front_left_camera_optical"
        pose.header.stamp = rospy.Time.now()
        pose.pose.position.x = position[0]
        pose.pose.position.y = position[1]
        pose.pose.position.z = position[2]
        try:
            pose_base_frame = self.tfBuffer.transform(pose, target_frame)
        except TransformException as e:
            print(f"In perception Module, cannot transform front_left_camera_optical to {target_frame}:")
            return
        x = pose_base_frame.pose.position.x
        y = pose_base_frame.pose.position.y
        z = pose_base_frame.pose.position.z
        min_distance = math.inf
        for digit, position in self.digits_and_positions:
            distance = np.linalg.norm([x - position[0], y - position[1]])
            if distance < min_distance:
                min_distance = distance
        if min_distance > BOX_SIZE:
            self.digits_and_positions.append((digit, (x, y)))
        self.visualize_marker(x, y, z, target_frame)

    def stereo_camera_callback(self, image: Image, disparity: DisparityImage):
        print("processing")
        if self.camera_info is None or not self.enable_signal:
            return
        image_array = self.cv_bridge.imgmsg_to_cv2(image, "bgr8")
        disparity_array = self.cv_bridge.imgmsg_to_cv2(disparity.image, desired_encoding="32FC1")
        digit_and_position = recognizer.recognize(PILImage.fromarray(image_array), detection_threshold=0.2,
                                                  area_threshold=600, debug=False)
        K = self.camera_info.K
        fx, fy = K[0], K[4]
        cx, cy = K[2], K[5]
        self.marker_array.markers.clear()
        for digit, position in digit_and_position:
            center_x, center_y = position
            z = (disparity.f * disparity.T) / disparity_array[center_y, center_x]
            x = (center_x - cx) * z / fx
            y = (center_y - cy) * z / fy
            self.merge_digit(digit, (x, y, z))
        self.marker_pub.publish(self.marker_array)
        print("---")

    def camera_info_callback(self, camera_info: CameraInfo):
        self.camera_info = camera_info


if __name__ == '__main__':
    rospy.init_node('perception', anonymous=True)
    counter = OnlineDigitCounter()
    rospy.spin()

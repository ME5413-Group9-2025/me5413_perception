# ROS
import rospy
from sensor_msgs.msg import Image, CameraInfo
from stereo_msgs.msg import DisparityImage
import cv_bridge
import message_filters
from tf2_ros import TransformListener, Buffer, TransformException
import tf2_geometry_msgs
from std_srvs.srv import SetBool, SetBoolResponse, SetBoolRequest
from visualization_msgs.msg import MarkerArray, Marker
from me5413_bringup.srv import SingleDigit, SingleDigitResponse
from std_msgs.msg import Int32MultiArray

import PIL.Image as PILImage
import math
import numpy as np
import sys
from pathlib import Path

parent_path = Path(__file__).parent.absolute()
sys.path.append(str(parent_path))

# Our module
import digit_recognizer as recognizer

# import minist.seg_pic

BOX_SIZE = 1


class OnlineDigitCounter:
    def __init__(self):
        self.camera_info = None
        self.left_image = None
        self.depth_image = None
        self.state = None
        self.enable_signal = True
        self.digits_and_positions = []
        self.marker_array = MarkerArray()
        self.cv_bridge = cv_bridge.CvBridge()
        self.tfBuffer = Buffer()
        self.listener = TransformListener(self.tfBuffer)
        self.left_image_sub = rospy.Subscriber("/front/left/image_raw", Image, self.left_image_callback)
        self.recognize_digit_server = rospy.Service("/recognize_target", SingleDigit, self.recognize_target_callback)
        self.enable_signal_server = rospy.Service("enable_perception_signal", SetBool, self.enable_signal_callback)
        self.left_camera_info_sub = rospy.Subscriber("/front/left/camera_info", CameraInfo, self.camera_info_callback)
        self.left_image_sub = message_filters.Subscriber("/front/left/image_raw", Image)
        self.disparity_sub = message_filters.Subscriber("/front/disparity", DisparityImage)
        self.sync = message_filters.TimeSynchronizer([self.left_image_sub, self.disparity_sub], queue_size=1)
        self.marker_pub = rospy.Publisher("/digit_markers", MarkerArray, queue_size=1)
        self.counter_result_pub = rospy.Publisher("/digit_count", Int32MultiArray, queue_size=1)
        self.sync.registerCallback(self.stereo_camera_callback)
        print("OnlineNumberCounter initialized")

    def enable_signal_callback(self, req: SetBoolRequest):
        self.enable_signal = req.data
        if not self.enable_signal:
            times = [0 for _ in range(10)]
            for digit, _ in self.digits_and_positions:
                times[digit] += 1
            topic = Int32MultiArray(data=times)
            self.counter_result_pub.publish(topic)
        return SetBoolResponse()

    def left_image_callback(self, image):
        self.left_image = image

    def recognize_target_callback(self, req):
        rospy.loginfo("Recognizing the target")
        image_array = self.cv_bridge.imgmsg_to_cv2(self.left_image, "bgr8")
        digit_and_position = recognizer.recognize(PILImage.fromarray(image_array), detection_threshold=0.05,
                                                  area_threshold=600, debug=False, return_max=True)
        if digit_and_position:
            response = SingleDigitResponse(digit=digit_and_position[0])
        else:
            rospy.logerr("Unable to recognize the target")
            response = SingleDigitResponse(digit=-1)
        return response

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
            pose_base_frame = self.tfBuffer.transform(pose, target_frame, timeout=rospy.Duration(nsecs=int(1e8)))
        except TransformException as e:
            rospy.logerr(e)
            rospy.logerr(f"In perception Module, cannot transform front_left_camera_optical to {target_frame}:")
            return
        x = pose_base_frame.pose.position.x
        y = pose_base_frame.pose.position.y
        z = pose_base_frame.pose.position.z
        min_distance = math.inf
        for digit, position in self.digits_and_positions:
            distance = np.linalg.norm([x - position[0], y - position[1]])
            if distance < min_distance:
                min_distance = distance
        if min_distance >= BOX_SIZE / 2:
            self.digits_and_positions.append((digit, (x, y)))
        self.visualize_marker(x, y, z, target_frame)

    def stereo_camera_callback(self, image: Image, disparity: DisparityImage):
        if self.camera_info is None or not self.enable_signal:
            return
        image_array = self.cv_bridge.imgmsg_to_cv2(image, "bgr8")
        disparity_array = self.cv_bridge.imgmsg_to_cv2(disparity.image, desired_encoding="32FC1")
        digit_and_position = recognizer.recognize(PILImage.fromarray(image_array), detection_threshold=0.2,
                                                  area_threshold=1000, debug=False)
        # digit_and_position = minist.seg_pic.recognize(PILImage.fromarray(image_array), detection_threshold=0.2,
        #                                               area_threshold=300, debug=False)
        self.marker_array.markers.clear()
        for digit, bbox in digit_and_position:
            position = recognizer.get_box_position(bbox, disparity_array, disparity.T, self.camera_info)
            if position is None:
                continue
            x, y, z = position
            self.merge_digit(digit, (x, y, z))
        self.marker_pub.publish(self.marker_array)

    def camera_info_callback(self, camera_info: CameraInfo):
        self.camera_info = camera_info


if __name__ == '__main__':
    rospy.init_node('perception', anonymous=True)
    counter = OnlineDigitCounter()
    rospy.spin()

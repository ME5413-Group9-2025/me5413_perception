import rospy
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped
from std_srvs.srv import SetBool, SetBoolResponse, SetBoolRequest


class BridgeDetector:
    def __init__(self):
        self.enable_signal = False
        self.map_sub = rospy.Subscriber('/map', OccupancyGrid, self.map_callback)
        self.pose_pub = rospy.Publisher('/bridge_pose', PoseStamped, queue_size=1)
        self.detect_service = rospy.Service('/enable_bridge_signal', SetBool, self.enable_detection_callback)
        self.map_data = None
        self.map_info = None
        self.bridge_pose = None
        self.bridge_coordinates = []

    def map_callback(self, msg):
        if not self.enable_signal:
            return
        self.map_data = msg.data
        self.map_info = msg.info
        self.detect_bridge()

    def enable_detection_callback(self, req: SetBoolRequest):
        self.enable_signal = req.data
        return SetBoolResponse(success=True)

    def detect_bridge(self):
        if self.map_data is None or self.map_info is None:
            return

        width = self.map_info.width
        resolution = self.map_info.resolution

        origin_x = self.map_info.origin.position.x
        origin_y = self.map_info.origin.position.y

        x_start = int(5 / resolution - origin_x / resolution)
        x_end = int(7 / resolution - origin_x / resolution)

        y_start = int(-23 / resolution - origin_y / resolution)
        y_end = int(-2 / resolution - origin_y / resolution)

        self.bridge_coordinates = []
        seen_y_coords = set()
        for x in range(x_start, x_end):
            for y in range(y_start, y_end):
                index = y * width + x
                if (self.map_data[index]) > 60:
                    coor_x = x * resolution + self.map_info.origin.position.x
                    coor_y = y * resolution + self.map_info.origin.position.y
                    if coor_y not in seen_y_coords:
                        self.bridge_coordinates.append((coor_x, coor_y))
                        seen_y_coords.add(coor_y)
                    

        if self.bridge_coordinates:
            y_values = [coord[1] for coord in self.bridge_coordinates]
            y_center = sum(y_values) / len(y_values)

            bridge_pose = PoseStamped()
            bridge_pose.header.frame_id = 'map'
            bridge_pose.header.stamp = rospy.Time.now()
            # yaw angle should be pi = -pi
            bridge_pose.pose.orientation.w = 0.0
            bridge_pose.pose.orientation.z = -1.0
            bridge_pose.pose.position.x = 9.5
            bridge_pose.pose.position.y = y_center
            self.pose_pub.publish(bridge_pose)
            rospy.loginfo(f"Published bridge center to /bridge_pose: (9.5, {y_center})")
            self.bridge_pose = bridge_pose
        else:
            rospy.logwarn("No bridge detected in the specified area!")


if __name__ == '__main__':
    rospy.init_node('bridge_detector', anonymous=True)
    detector = BridgeDetector()
    rospy.spin()
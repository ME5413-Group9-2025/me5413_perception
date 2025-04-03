#!/usr/bin/env python
import rospy
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Point
import cv2 as cv
import numpy as np
from PIL import Image, ImageDraw

class BridgeDetector:
    def __init__(self):
        # 订阅地图话题
        self.map_sub = rospy.Subscriber('/map', OccupancyGrid, self.map_callback)
        # 创建发布者来发布桥的中心点
        self.point_pub = rospy.Publisher('/bridge_center', Point, queue_size=10)
        # 创建服务来触发检测
        self.detect_service = rospy.Service('/detect_bridge', Empty, self.handle_detect_bridge)


        self.map_data = None
        self.map_info = None
        self.bridge_position = None
        self.bridge_coordinates = []
        
        # # 添加一个标志来控制是否已经完成检测
        # self.detection_complete = False
        # # 添加一个计时器来延迟检测
        # self.last_map_time = None
        # self.DELAY_THRESHOLD = 0.5  # 等待0.5秒没有新地图消息时才进行检测

    def map_callback(self, msg):
        # 保存地图数据
        self.map_data = msg.data
        self.map_info = msg.info

        # self.last_map_time = rospy.get_time()

        # # 如果检测尚未完成，启动计时器
        # if not self.detection_complete:
        #     rospy.Timer(rospy.Duration(self.DELAY_THRESHOLD), self.check_and_detect, oneshot=True)


    # def check_and_detect(self, event):
    #     # 检查是否超过延迟时间且没有新消息
    #     current_time = rospy.get_time()
    #     if self.last_map_time and (current_time - self.last_map_time >= self.DELAY_THRESHOLD):
    #         if not self.detection_complete:
    #             self.detect_bridge()
    #             self.detection_complete = True  # 标记检测完成，避免重复检测

    def handle_detect_bridge(self, req):
        # 服务回调函数，当被调用时执行检测
        if self.map_data is None or self.map_info is None:
            rospy.logwarn("No map data available yet! Please wait for map to be published.")
            return EmptyResponse()
        
        self.detect_bridge()
        return EmptyResponse()

    def detect_bridge(self):
        if self.map_data is None or self.map_info is None:
            return
        
        map_data = np.reshape(self.map_data, (self.map_info.height, self.map_info.width))
        map_data = np.array(map_data, dtype=np.int8)
        map_data = np.where(map_data == -1, 0, map_data).astype(np.uint8)
        image = Image.fromarray(map_data, mode="L")        

        width = self.map_info.width
        height = self.map_info.height
        resolution = self.map_info.resolution
        
        origin_x = self.map_info.origin.position.x
        origin_y = self.map_info.origin.position.y


        # draw = ImageDraw.Draw(image)
        # center_x = width - int(origin_x / resolution)
        # center_y = height - int(origin_y / resolution)
        # radius = 100
        # left = center_x - radius
        # top = center_y - radius
        # right = center_x + radius
        # bottom = center_y + radius

        x_start = int(5/resolution-self.map_info.origin.position.x/resolution)
        x_end = int(7/resolution-self.map_info.origin.position.x/resolution)

        y_start = int(-23/resolution-self.map_info.origin.position.y/resolution)
        y_end = int(-2/resolution-self.map_info.origin.position.y/resolution)


        # for y in range(y_start, y_end):
        #     if self.map_data[3 * height +y] == 0:
        #         rospy.loginfo(f"Bridge detected")
        self.bridge_coordinates = []

        for x in range(x_start, x_end):
            for y in range(y_start, y_end):
                index = y * width + x
                # print(f"({x}, {y})")




                # print(f"({x}, {y}): {self.map_data[index]}")

                # rospy.loginfo(self.map_data[x * height +y])
                if(self.map_data[index]) > 60:
                    # print(f"Bridge detected at ({x}, {y})")
                    coor_x= x * resolution + self.map_info.origin.position.x
                    coor_y= y * resolution + self.map_info.origin.position.y
                    self.bridge_coordinates.append((coor_x, coor_y))
                    # print(f"Coordinate: ({coor_x}, {coor_y})")

        # 计算y轴中心值
        if self.bridge_coordinates:
            # 提取所有y坐标并计算平均值
            y_values = [coord[1] for coord in self.bridge_coordinates]
            y_center = sum(y_values) / len(y_values)
            # print(f"center of bridge: 8.5, {y_center}")
            
            # 创建并发布点(8.5, y_center)
            bridge_point = Point()
            bridge_point.x = 8.5
            bridge_point.y = y_center
            bridge_point.z = 0.0  # z设为0，因为我们只关心2D位置
            self.point_pub.publish(bridge_point)
            rospy.loginfo(f"Published bridge center to /bridge_center: (8.5, {y_center})")

            self.bridge_position = bridge_point
        else:
            rospy.logwarn("No bridge detected in the specified area!")


        

    def get_bridge_position(self):
        return self.bridge_position

if __name__ == '__main__':
    rospy.init_node('bridge_detector', anonymous=True)
    detector = BridgeDetector()
    rospy.spin()



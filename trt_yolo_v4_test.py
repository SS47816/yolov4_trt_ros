#!/usr/bin/env python

import os
import time
import math
import numpy as np
import sys

import cv2
import pycuda.autoinit  # For initializing CUDA driver
import pycuda.driver as cuda
import pyzed.sl as sl # For Zed SDK

from utils.yolo_classes import get_cls_dict, CLASSES_LIST
from utils.display import open_window, set_display, show_fps
from utils.visualization import BBoxVisualization
from utils.yolo_with_plugins import TrtYOLO

import rospy
import rospkg
from yolov4_trt_ros.msg import Detector2DArray
from yolov4_trt_ros.msg import Detector2D
from vision_msgs.msg import BoundingBox2D
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2
from cv_bridge import CvBridge, CvBridgeError
from autoware_msgs.msg import DetectedObject, DetectedObjectArray


class yolov4(object):
    def __init__(self):
        """ Constructor """

        self.bridge = CvBridge()
        self.init_params()
        self.init_yolo()
        self.cuda_ctx = cuda.Device(0).make_context()
        self.trt_yolo = TrtYOLO(
            (self.model_path + self.model), (self.h, self.w), self.category_num)
        self.init_zed()

    def __del__(self):
        """ Destructor """

        self.cuda_ctx.pop()
        del self.trt_yolo
        del self.cuda_ctx

    def clean_up(self):
        """ Backup destructor: Release cuda memory """

        if self.trt_yolo is not None:
            self.cuda_ctx.pop()
            del self.trt_yolo
            del self.cuda_ctx

    def init_params(self):
        """ Initializes ros parameters """
        
        rospack = rospkg.RosPack()
        package_path = rospack.get_path("yolov4_trt_ros")
        self.video_topic = rospy.get_param("/video_topic", "/zed_node/left/image_rect_color")
        self.model = rospy.get_param("/model", "yolov4")
        self.model_path = rospy.get_param("/model_path", package_path + "/yolo/")
        self.category_num = rospy.get_param("/category_number", 80)
        self.input_shape = rospy.get_param("/input_shape", "416")
        self.conf_th = rospy.get_param("/confidence_threshold", 0.5)
        self.show_img = rospy.get_param("/show_image", False)

        # self.image_sub = rospy.Subscriber(
        #     self.video_topic, Image, self.img_callback, queue_size=1, buff_size=1280*720*3)
        self.img_raw_pub = rospy.Publisher(
            "/detection/yolov4_detector/image_raw", Image, queue_size=1)
        self.img_result_pub = rospy.Publisher(
            "/detection/yolov4_detector/image_results", Image, queue_size=1)
        self.pointcloud_pub = rospy.Publisher(
            "/detection/yolov4_detector/pointcloud", PointCloud2, queue_size=1)
        self.detection_pub_autoware = rospy.Publisher(
            "/detection/yolov4_detector/objects", DetectedObjectArray, queue_size=1)
    
    def init_yolo(self):
        """ Initialises yolo parameters required for trt engine """

        if self.model.find('-') == -1:
            self.model = self.model + "-" + self.input_shape
            
        yolo_dim = self.model.split('-')[-1]

        if 'x' in yolo_dim:
            dim_split = yolo_dim.split('x')
            if len(dim_split) != 2:
                raise SystemExit('ERROR: bad yolo_dim (%s)!' % yolo_dim)
            self.w, self.h = int(dim_split[0]), int(dim_split[1])
        else:
            self.h = self.w = int(yolo_dim)
        if self.h % 32 != 0 or self.w % 32 != 0:
            raise SystemExit('ERROR: bad yolo_dim (%s)!' % yolo_dim)

        cls_dict = get_cls_dict(self.category_num)
        self.vis = BBoxVisualization(cls_dict)

    def init_zed(self):
        """ Initialises zed camera and continuously capture images and do object detection """
        # Create a ZED camera object
        self.zed = sl.Camera()

        # Set configuration parameters
        zed_init_params = sl.InitParameters()
        zed_init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE  # Use PERFORMANCE depth mode
        zed_init_params.coordinate_units = sl.UNIT.METER        # Use meter units (for depth measurements)
        zed_init_params.camera_resolution = sl.RESOLUTION.HD720
        zed_init_params.camera_fps = 30

        # Open the camera
        err = self.zed.open(zed_init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            exit(1)
        
        # Create and set RuntimeParameters after opening the camera
        runtime_parameters = sl.RuntimeParameters()
        runtime_parameters.sensing_mode = sl.SENSING_MODE.STANDARD  # Use STANDARD sensing mode
        # Setting the depth confidence parameters
        runtime_parameters.confidence_threshold = 100
        runtime_parameters.textureness_confidence_threshold = 100

        # Grabing images
        image = sl.Mat()
        depth = sl.Mat()
        point_cloud = sl.Mat()
        mirror_ref = sl.Transform()
        mirror_ref.set_translation(sl.Translation(2.75,4.0,0))
        tr_np = mirror_ref.m

        while True:
            # Grab an image, a RuntimeParameters object must be given to grab()
            if self.zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                # A new image is available if grab() returns SUCCESS
                # Retrieve left image
                self.zed.retrieve_image(image, sl.VIEW.LEFT)
                # Retrieve depth map. Depth is aligned on the left image
                self.zed.retrieve_measure(depth, sl.MEASURE.DEPTH, sl.MEM.GPU)
                # Retrieve colored point cloud. Point cloud is aligned on the left image.
                self.zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.GPU)

                timestamp = self.zed.get_timestamp(sl.TIME_REFERENCE.CURRENT)  # Get the timestamp at the time the image was captured
                print("Image resolution: {0} x {1} || Image timestamp: {2}\n".format(image.get_width(), image.get_height(),
                    timestamp.get_milliseconds()))

                self.publish_img(self, self.img_raw_pub, image)
                self.publish_img(self, self.img_result_pub, depth)
                print("Images published")

                # Get and print distance value in mm at the center of the image
                # We measure the distance camera - object using Euclidean distance
                x = round(image.get_width() / 2)
                y = round(image.get_height() / 2)
                err, point_cloud_value = point_cloud.get_value(x, y)

                distance = math.sqrt(point_cloud_value[0] * point_cloud_value[0] +
                                    point_cloud_value[1] * point_cloud_value[1] +
                                    point_cloud_value[2] * point_cloud_value[2])

                point_cloud_np = point_cloud.get_data()
                point_cloud_np.dot(tr_np)

                if not np.isnan(distance) and not np.isinf(distance):
                    # print("Distance to Camera at ({}, {}) (image center): {:1.3} m".format(x, y, distance), end="\r")
                    # Increment the loop
                    i = i + 1
                else:
                    print("Can't estimate distance at this position.")
                    print("Your camera is probably too close to the scene, please move it backwards.\n")
                sys.stdout.flush()

        self.zed.close()
                

    def img_callback(self, ros_img):
        """Continuously recieve images from ros topic and do object detection """

        tic = time.time()

        # converts from ros_img to cv_img for processing
        try:
            cv_img = self.bridge.imgmsg_to_cv2(
                ros_img, desired_encoding="bgr8")
            rospy.logdebug("ROS Image converted for processing")
        except CvBridgeError as e:
            rospy.loginfo("Failed to convert image %s", str(e))

        if cv_img is not None:
            boxes, confs, clss = self.trt_yolo.detect(cv_img, self.conf_th)

            cv_img = self.vis.draw_bboxes(cv_img, boxes, confs, clss)
            toc = time.time()
            fps = 1.0 / (toc - tic)

            self.publisher_autoware(boxes, confs, clss)

            if self.show_img:
                cv_img = show_fps(cv_img, fps)
                cv2.imshow("YOLOv4 DETECTION RESULTS", cv_img)
                cv2.waitKey(1)

        # converts back to ros_img type for publishing
        try:
            overlay_img = self.bridge.cv2_to_imgmsg(
                cv_img, encoding="passthrough")
            rospy.logdebug("CV Image converted for publishing")
            overlay_img.header.frame_id = 'left_cam' # change accordingly
            overlay_img.encoding = 'bgr8'
            self.img_result_pub.publish(overlay_img)
        except CvBridgeError as e:
            rospy.loginfo("Failed to convert image %s", str(e))

    def publish_img(self, publisher, cv_img):
        """ Convert the cv_img to ros_img and publish it """

        try:
            img = self.bridge.cv2_to_imgmsg(cv_img, encoding="passthrough")
            rospy.logdebug("CV Image converted for publishing")
            img.header.frame_id = 'left_cam' # change accordingly
            img.encoding = 'bgr8'
            publisher.publish(img)
        except CvBridgeError as e:
            rospy.loginfo("Failed to convert image %s", str(e))


    def publisher_autoware(self, boxes, confs, clss):
        """ Publishe the results to autoware_msgs

        `Parameters`
        boxes (List(List(int))) : Bounding boxes of all objects

        confs (List(double))	: Probability scores of all objects

        clss  (List(int))	: Class ID of all classes
        """

        detection2d = DetectedObjectArray()
        
        # detection2d.header.stamp = rospy.Time.now()
        # detection2d.header.frame_id = "usb_cam" # change accordingly

        for i in range(len(boxes)):
            # boxes : xmin, ymin, xmax, ymax

            # for _ in boxes:
            detection = DetectedObject()
            detection.header.stamp = rospy.Time.now()
            detection.header.frame_id = "zed_camera_center" # change accordingly
            detection.id = clss[i]
            detection.score = confs[i]
            detection.label = CLASSES_LIST[int(clss[i])]

            detection.x = boxes[i][0]
            detection.y = boxes[i][1]
            detection.width = abs(boxes[i][0] - boxes[i][2])
            detection.height = abs(boxes[i][1] - boxes[i][3])

            detection2d.objects.append(detection)
        
        self.detection_pub_autoware.publish(detection2d)


def main():
    yolo = yolov4()
    rospy.init_node('yolov4_trt_ros', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.on_shutdown(yolo.clean_up())
        print("Shutting down")


if __name__ == '__main__':
    main()

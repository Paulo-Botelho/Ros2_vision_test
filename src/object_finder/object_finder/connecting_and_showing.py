#!/usr/bin/env python3
# coding=utf-8

import rclpy
from rclpy.node import Node


#from sensor_msgs.msg import Image as ROS_Image

import cv2
#from cv_bridge import CvBridge
from .running_inference import running_inference as ri


from vision_msg.msg import Ball
#from vision_msgs.msg import Webotsmsg
import sys

sys.setrecursionlimit(100000)
width = 416 # Largura da imagem (conferir no vídeo)
height = 416 # Altura da imagem (Conferir no vídeo)

'''import cProfile, pstats, io
from pstats import SortKey
pr = cProfile.Profile()
pr.enable()
'''

class VisionNode(Node):
        #Init
    def __init__(self,nome_no):
        super().__init__('no_visao')
        
        #Iniciando o ROS
        #Capturar parametros (qual camera e se queremos output de imagem) do launch
        self.camera = 0#rospy.get_param('vision/camera')
        self.output_img = True#rospy.get_param('vision/img_output')
        self.ajuste = False#rospy.get_param('vision/ajuste')
        self.bright = 4#rospy.get_param('vision/brilho')


        #Iniciando o nó e obtendo os arquivos que definem a rede neural
        #rospy.init_node(nome_no, anonymous = True)
        self.net = ri.get_cnn_files()
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)   
        self.model = ri.set_model_input(self.net)
        self.searching = True
        self.cap = cv2.VideoCapture(self.camera,cv2.CAP_ANY)
        self.cap.set(cv2.CAP_PROP_BRIGHTNESS, (self.bright))
        self.publisher_ = self.create_publisher(Ball, '/bioloid/vision_inference',  10)
        #self.publisher = rospy.Publisher('/webots_natasha/vision_inference', Webotsmsg, queue_size=100)

        
        #SE FOR NO REAL
        self.get_webcam()

        #SE FOR NO WEBOTS
        #self.connect_to_webots()
        
        


    def get_webcam(self):

        print("\n----Visão Operante----\n")
        if self.ajuste == True:
            print("Ajuste de Brilho '=' para aumentar e '-' para diminuir.\n")
            print("Para continuar a detecção. Aperte W.\n")

        while True:
            ret , self.current_frame = self.cap.read()
            self.classes, self.scores, self.boxes, self.fps = ri.detect_model(self.model,self.current_frame)

            if not ret:
                print("Error capturing frame")
                break

            if self.output_img == True:
                self.show_result_frame()

            if self.ajuste == True:
                self.ajuste_camera()
            

            self.publish_results()

            if cv2.waitKey(1) == ord("q") :
                self.cap.release()
                cv2.destroyAllWindows()


    def show_result_frame(self):
        '''Shows the result frame obtained from neural network on OpenCV window.'''
        ri.draw_results(self.current_frame, self.classes, self.scores, self.boxes)
        cv2.imshow("Current Frame", self.current_frame)


        
    def publish_results(self):

        objects_msg = Ball()

        for i in range(len(self.boxes)):
            [x_top, y_top, roi_width, roi_height] = self.boxes[i]

            x = int(x_top + roi_width/2)
            y = int(y_top + roi_height/2)
            
            results = [True, x, y, int(roi_width), int(roi_height)]

    

            if self.classes[i]== 1:

                objects_msg.found, objects_msg.x, objects_msg.y, objects_msg.roi_width, objects_msg.roi_height = results

        self.publisher_.publish(objects_msg)


    def connect_to_webots(self):
        '''Gets the Vision topic sent from Behavior, and subscribe it.'''

        self.topic_found = False
        while self.topic_found == False:
            try:
                for sublist in rospy.get_published_topics(namespace = "/"):
                    for item in sublist:
                        if "vision_controller" in item:
                            self.vision_topic = item

                rospy.Subscriber(self.vision_topic, ROS_Image, callback = self.convert_ros_image_to_cv2)
                self.topic_found = True
                rospy.spin()
            except Exception:
                pass
            
    def convert_ros_image_to_cv2(self, message):
        '''Converts the sensor_msgs/Image to Numpy Array'''

        self.opencv_bridge = CvBridge()
        
        try:
            self.current_frame = self.opencv_bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
        
        except Exception as e:
            print(f"{e}")

        self.send_current_frame_to_inference()
        #Diferencias códigos da camera e do Webots

    #Configurações da imagem (Brilho) (Parametro passado launch)

    def ajuste_camera(self):
        
        while cv2.waitKey(1) != ord("w"):
            
            if cv2.waitKey(1) == ord('='):
                self.bright = self.bright + 10
                if self.cap.get(cv2.CAP_PROP_BRIGHTNESS) < 64:
                    self.cap.set(cv2.CAP_PROP_BRIGHTNESS, (self.bright))
                else:
                    self.bright = 64                
                print("Brightness property current value:", self.cap.get(cv2.CAP_PROP_BRIGHTNESS))

            if cv2.waitKey(1) == ord('-'):
                self.bright = self.bright - 10
                if self.cap.get(cv2.CAP_PROP_BRIGHTNESS) > -64:
                    self.cap.set(cv2.CAP_PROP_BRIGHTNESS, (self.bright))
                else:
                    self.bright = -64     
                print("Brightness property current value:", self.cap.get(cv2.CAP_PROP_BRIGHTNESS))

            #Atualizar Frame
            _ , self.current_frame = self.cap.read()
            cv2.imshow("Current Frame", self.current_frame)
        self.ajuste = False
        



'''
pr.disable()
s = io.StringIO()
sortby = SortKey.CUMULATIVE
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats()
print(s.getvalue())'''

def main(args=None):
    rclpy.init(args=args)
    no_visao = VisionNode('visao')
    rclpy.spin(no_visao)    
    no_visao.destroy_node()    
    rclpy.shutdown()


if __name__ == '__main__':
    main()

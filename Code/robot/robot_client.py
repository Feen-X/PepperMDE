# coding: utf-8
#
# Copyright 2021 The Technical University of Denmark
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from threading import Thread
import paramiko
from scp import SCPClient
import math
import sys

import cv2
import numpy as np
import qi
import time
import json
import re

def print_debug(*args, **kwargs):
    """
    Print a debug message to stderr.
    """
    print(*args, file=sys.stderr, **kwargs)

qi.logging.setLevel(qi.logging.ERROR)


# This is the map of ip addresses to port numbers for the server, based on the ip address of the robot.
# The IP can change, but the port should not. If the IP changes, the port number should be updated in the server code.


class RobotClient:
    def __init__(self, ip: str, config_file: str = "robot/robot_config.json"):
        self.ip: str = ip
        self.username: str = None
        self.password: str = None
        self.port: int = None
        self.config_file = config_file

        self.angle = 0

        self.__load_config()
        ssh = self.__connect_to_robot_SSH()
        self.scp = SCPClient(ssh.get_transport())

        self.__initialize_ALProxies()
        self.__initialize_robot()

        # Mapping of the directions to the angles (use these to turn the robot in the right direction i.e. the shortest path to the direction)
        self.direction_mapping = {
            'Move(N)': 90,
            'Move(E)': 0,
            'Move(S)': 270,
            'Move(W)': 180,
        }

    def __load_config(self):
        """
        Load the robot configuration from the robot config file.
        The configuration file contains the robot configuration such as the username, password, port and vision port.
        """
        # Load the configuration file
        try:
            with open(self.config_file, 'r') as config_file:
                config = json.load(config_file)
        except Exception as e:
            raise Exception(
                "Could not load robot configuration - file not found.", e)

        if self.ip not in config:
            raise Exception(
                "Robot's IP not in configuration file, please update the configuration file with the correct robot IP.")

        robot_config = config.get(self.ip)

        self.username = robot_config['username']
        self.password = robot_config['password']
        self.port = int(robot_config['port'])

    def __connect_to_robot_SSH(self) -> paramiko.SSHClient:
        """
        Connect to the robot using SSH, and also make a separate connection for the vision stream.
        """
        try:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.load_system_host_keys()
            ssh.connect(self.ip, username=self.username,
                        password=self.password)
            return ssh
        except paramiko.AuthenticationException:
            print_debug(
                "Authentication failed, please verify your credentials.")
            raise

    def __initialize_ALProxies(self):
        """Lazy initialization of ALProxies."""

        self.session = qi.Session()

        try:
            self.session.connect("tcp://" + self.ip + ":9559")
        except RuntimeError:
            raise Exception("Can't connect to Naoqi at ip \"" +
                            self.ip + "\" on port 9559.")

        self.tts = self.session.service("ALTextToSpeech")
        self.motion = self.session.service("ALMotion")
        # self.behavior = self.session.service("ALBehaviorManager")
        # self.tracker = self.session.service("ALTracker")
        self.posture = self.session.service("ALRobotPosture")
        # self.mem = self.session.service("ALMemory")
        # self.asr = self.session.service("ALSpeechRecognition")
        # self.leds = self.session.service("ALLeds")
        self.video = self.session.service("ALVideoDevice")
        # self.recorder = self.session.service("ALAudioRecorder")
        self.player = self.session.service("ALAudioPlayer")
        # self.tablet = self.session.service("ALTabletService")
        self.system = self.session.service("ALSystem")
        self.pm = self.session.service("ALPreferenceManager")
        # self.touch = self.session.service("ALTouch")

    def __initialize_robot(self):
        """
        Initialize the robot to a server pose.
        """
        # Setting collision protection False (will interfere with motion based ALProxies if True)
        self.motion.setExternalCollisionProtectionEnabled("Move", False)
        self.motion.setExternalCollisionProtectionEnabled("Arms", False)

        # Wake up robot (if not already up) and go to standing posture)
        self.motion.wakeUp()
        self.posture.goToPosture("Stand", 0.5)

        # Robot anounces that it is ready
        # self.say("I am ready")

        # Print the robot's IP and vision status to the console
        print_debug('Robot with IP: %r initialized' %
                    (self.ip))

    def forward(self, distance: float, block: bool = True) -> None:
        """
        Commands the robot to move forward a given distance in meters. 

        Parameters
        ----------
        'distance' : float
            The distance to move forward in meters.
        'block' : bool
            If true, the robot will wait until the motion is completed before continuing with the next command.
            If false, the robot will continue immediately. 
        """
        if block == False:
            Thread(target=(lambda: self.motion.moveTo(distance, 0, 0))).start()
        else:
            self.motion.moveTo(distance, 0, 0)

    def backward(self, distance: float, block: bool = True) -> None:
        self.forward(-distance, block)

    def stop(self) -> None:
        """
        Commands the robot to stop all motion immediately.
        """
        self.motion.stopMove()
    
    def move(self, new_dir: str) -> None:
        if str(new_dir) == 'NoOp': return
        current_angle = self.angle
        self.angle = self.direction_mapping[str(new_dir[0])]
        turn = self.angle - current_angle
        if turn > 180:
            self.turn_clockwise(math.radians(360-turn))
        elif turn < -180:
            self.turn_clockwise(math.radians(-360-turn))
        else:
            self.turn_clockwise(math.radians(-turn))
        self.forward(0.55)

    def push(self, x, _ = None):
        self.move(x)
        self.forward(0.2)
        self.backward(0.2)

    def say(self, sentence: str, language: str = "English") -> None:
        """
        Commands the robot to speak out the given sentence.

        The speech is generated using the onboard text-to-speech synthesis.
        Its intonation can sometimes be a bit strange. It is often possible to improve the
        understandability of the speech by inserting small breaks a key location in the sentence.
        This can be accomplished by inserting \\pau=$MS\\ commands, where $MS is the length of the
        break in milliseconds, e.g., robot.say("Hello \\pau=500\\ world!") will cause the robot to
        pause for 500 milliseconds before continuing with the sentence.

        Parameters
        ----------
        'sentence' : string
            The sentence to be spoken out loud.
        """
        Thread(target=(
            lambda: self.tts.say(sentence) if language == "English" else self.tts.say(sentence, language))).start()

    def turn_counter_clockwise(self, angle: float, block: bool = True) -> None:
        """
        Commands the robot to turn around its vertical axis.

        The position of the robot will remain approximately constant during the motion.
        Expect that the actually turned angle will vary a few degrees from the commanded values.
        The speed of the motion will be determined dynamically, i.e., the further it has to turn,
        the faster it will move.

        Parameters
        ----------
        'angle' : float
            The angle to turn in radians in the counter-clockwise direction.
        """
        if not block:
            Thread(target=(lambda: self.motion.moveTo(0, 0, angle))).start()
        else:
            self.motion.moveTo(0, 0, angle)

    def turn_clockwise(self, angle: float, block: bool = True) -> None:
        self.turn_counter_clockwise(-angle, block)

    def stand(self) -> None:
        """
        Commands the robot to stand up in a straight position. This should be called often to ensure the actuators dont overhead and are not damaged.
        """
        self.posture.goToPosture("Stand", 0.5)

    def head_position(self, yaw: float, pitch: float, relative_speed: float = 0.05) -> None:
        """
        Commands the robot to move its head to a specific position.

        The head can be moved in two directions: yaw and pitch.

        Parameters
        ----------
        'yaw' : float
            The angle to move the head in the horizontal direction. 
            Must be in range [-2.0857, 2.0857] radians or [-119.5, 119.5] degrees.
        'pitch' : float
            The angle to move the head in the vertical direction.
            Must be in range [-0.7068, 0.6371] radians or [-40.5, 36.5] degrees.
        'relative_speed' : float
            The relative max speed of the head motion. 
            Must be in range [0, 1]. 
            Avalue of 1.0 will move the head as fast as possible, while a value of 0 will move the head as slow as possible.
        """
        self.motion.setStiffnesses("Head", 1.0)
        self.motion.setAngles(["HeadYaw", "HeadPitch"], [
                              yaw, pitch], relative_speed)


    def download_file(self, file_name):
        """
        Download a file from robot to ./tmp folder in root.
        ..warning:: Folder ./tmp has to exist!
        :param file_name: File name with extension (or path)
        :type file_name: string
        """
        self.scp.get(file_name, local_path="tmp/")
        print_debug("[INFO]: File tmp/" + file_name + " downloaded")
        self.scp.close()


    def start_camera(self, camera_id=0, resolution=1):
        """
        Start Pepper camera streaming in a background thread.
        resolution: 0=160x120, 1=320x240, 2=640x480
        """
        self.camera_thread = PepperCameraThread(
            self.video,
            camera_id=camera_id,
            resolution=resolution,
        )
        self.camera_thread.start()
        print_debug("Camera stream started.")

    def stop_camera(self):
        if hasattr(self, "camera_thread") and self.camera_thread:
            self.camera_thread.stop()
            print_debug("Camera stream stopped.")

    @property
    def video_frame(self):
        """
        Returns the latest camera frame (np.ndarray) or None.
        """
        if hasattr(self, "camera_thread"):
            return self.camera_thread.frame
        return None


    def shutdown(self):
        self.stop_camera()
        self.motion.rest()


class PepperCameraThread(Thread):
    def __init__(self, video_service, camera_id=0, resolution=1):
        super().__init__()
        self.video_service = video_service
        self.camera_id = camera_id
        self.resolution = resolution
        self.running = False
        self.frame = None

        self.client_name = self.video_service.subscribeCamera(
            "robot_client_faststream",
            camera_id,
            resolution,
            13,    # BGR
            30     # let Pepper decide maximum FPS
        )

    def run(self):
        self.running = True
        while self.running:
            img = self.video_service.getImageRemote(self.client_name)
            if img:
                w, h = img[0], img[1]
                data = img[6]
                arr = np.frombuffer(data, dtype=np.uint8)
                self.frame = arr.reshape((h, w, 3))
            time.sleep(0.0001)

    def stop(self):
        self.running = False
        time.sleep(0.05)
        self.video_service.unsubscribe(self.client_name)



if __name__ == '__main__':
    # get the ip address of the robot
    ip = sys.argv[1]

    # create a robot object and pass the ip address, and set vision to True (set to False, if you don't want to use the vision)
    robot = RobotClient(ip, vision=True)

    robot.say("Hello, I am Pepper!")

    robot.forward(0.2)
    robot.backward(0.2)

    robot.turn_counter_clockwise(math.radians(45))
    robot.turn_clockwise(math.radians(45))

    robot.head_position(-1.0, 0.0)
    time.sleep(3)
    robot.head_position(0.0, 0.0)

    robot.stand()

    robot.say("Listening")
    robot.listen(5, playback=True)

    robot_vision = robot.instantiate_vision_processes(ip, robot.vision_port)

    # Wait for vision thread to start
    while robot_vision.frame is None:
        print_debug("Waiting for vision thread to start...")
        time.sleep(1)

    # Look for tags in the image
    robot.head_position(0, math.radians(22.5), relative_speed=0.1)
    if robot_vision.tag_in_view:
        print_debug("Closest tag to the bottom middle of the image:",
              robot_vision.closest_tag)
        print_debug("Middle bottom of the image:", robot_vision.middle_bottom)
    else:
        print_debug("No tag in view")

    try:
        while True:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

            if robot_vision.frame is None:
                continue

            cv2.imshow(f"Robot {ip} Vision", robot_vision.frame['image'])

            if robot_vision.tag_in_view:
                target_center = robot_vision.closest_tag['tag_center']
                print("Tag Center: ", target_center)

    except KeyboardInterrupt:
        pass

    robot_vision.stop()
    robot.shutdown()

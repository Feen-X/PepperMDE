from robot_client import RobotClient
import sys
import cv2
import time

if __name__ == "__main__":
    try:
        ip = sys.argv[1]
    except IndexError:
        # ip = "192.168.1.109"
        ip = "192.168.1.113"

    robot = RobotClient(ip)

    # Get starting head position
    yaw, pitch = robot.motion.getAngles(["HeadYaw", "HeadPitch"], True)

    robot.start_camera(camera_id=0, resolution=2)

    cv2.namedWindow("Pepper Camera", cv2.WINDOW_NORMAL)

    # print("Use WASD to move head, Q to quit.")

    start = time.time()
    robot.say("Streaming")
    while True:
        # ---- CAMERA STREAM ----
        frame = robot.video_frame
        if frame is not None:
            w, h = frame.shape[1], frame.shape[0]
            bigger = cv2.resize(frame, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
            cv2.imshow("Pepper Camera", bigger)

        # ---- READ KEY (non-blocking) ----
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            robot.shutdown()
            break
        elif key == ord('p'):
            break
            
        incr = 0.2
        # ---- CONTINUOUS CONTROL ----
        if key == ord('w'):      # Up
            pitch -= incr
        elif key == ord('s'):    # Down
            pitch += incr
        elif key == ord('a'):    # Left
            yaw += incr
        elif key == ord('d'):    # Right
            yaw -= incr
        elif key == ord(' '):    # Center
            yaw, pitch = 0.0, 0.0
            
        # Clamp values to avoid extreme positions
        yaw = max(-2.0857, min(2.0857, yaw))
        pitch = max(-0.7068, min(0.6371, pitch))

        # ---- MOVE HEAD USING head_position() ----
        robot.head_position(yaw, pitch)
        
    robot.stop_camera()
    cv2.destroyAllWindows()

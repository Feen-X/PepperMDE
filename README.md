<div align="center">
  <h1>Monocular Depth Estimation in Robotics</h1>
  <h2>From Survey to Perception–Action with the Pepper Robot</h2>
  <p>Author: Nicklas Mundt</p>
  <p>Date: October 2025</p>
</div>
This repository contains all the relevant files for my Bachelor Project. This includes the code, report, experiments and more. The project focuses on monocular depth estimation and tries to illustrate its potential in robotics by using the Pepper Robot.

<p align="center">
  <img src="Report/Experiment/PepperHi.gif" alt="Pepper robot navigating an environment" width="60%"/>
</p>
  
## Project Structure
This project is structured into two main directories: `Code/` and `Report/`. The `Code/` directory contains all the source code for depth estimation, object detection, and robot integration, as well as the main file for the Perception-Action system. The `Report/` directory contains the project report, experiment documentation, and related files and images.
```
PepperMDE/
├── Code/
|   ├── .devcontainer/
|   ├── Images/
|   ├── DepthEstimation/
|   ├── ObjectDetection/
|   ├── robot/
|   └── final_main.py
├── Report/
|   ├── Experiment/
|   ├── Project_Plan/
|   ├── Report_images/
|   └── Report.pdf
├── .gitignore
└── README.md
```
## Requirements for the Code
To run the code, you will need to have [Docker](https://www.docker.com/products/docker-desktop/) installed on your machine. The project uses a Docker container to ensure a consistent environment across different systems. The Dockerfile is located in the `.devcontainer/` directory within the `Code/` folder.

You also need to download a pre-trained YOLO model for object detection. For this project, the model `yolov11l.pt` was used, which you can find on the [YOLO website](https://docs.ultralytics.com/models/yolo11/). This model should be placed in the `Code/ObjectDetection/models/` directory (create the folder if it does not exist).

Finally, ensure that the correct configurations are set in the `final_main.py` file, such as the YOLO model path, Pepper robot IP address, and any other parameters you wish to adjust (they are adjusted in the top of the file).

## Getting Started
To get started with the Perception-Action system, clone the entire repository and navigate to the `Code/` directory
```bash
git clone https://github.com/Feen-X/PepperMDE.git
cd PepperMDE/Code
```
From here, you can either explore the individual modules for depth estimation and object detection (they have their own example scripts), or run the `final_main.py` script to see the integrated system in action. Before running the main script, check if the `IMG_MODE` variable is set to your desired mode of operation (can be changed in the top of the script together with the other parameters). If `True`, the system will use images from the `Images/` directory for testing. If `False`, it will connect to the Pepper robot's camera feed, with the specified settings.

To run the main script, first activate the Docker container (if using VSCode, open the folder in a Dev Container), then execute:
```bash
python final_main.py
```
The first time you run the code, the monocular depth estimation model will be downloaded automatically, which may take some time depending on your internet connection. Once downloaded, all subsequent runs will be faster.

When the script is running, you can observe the output in the browser window by navigating to `http://localhost:8080/vnc.html`. Here, you will see the camera feed with detected objects and their bounding boxes, as well as the estimated depth map, depending on which state the system is in. The current state is displayed in the upper part of the window.

## Running an experimental trial with Pepper
To run an experimental trial with the Pepper robot, ensure that the robot is powered on and connected to the same network as your development machine. Update the `PEPPER_IP` variable in `final_main.py` with the robot's IP address. Also, set the `IMG_MODE` variable to `False` to enable live camera feed from Pepper. Once everything is set up, when you run the `final_main.py` script, the robot will start capturing images, performing object detection and depth estimation, and executing the defined actions based on the perceived environment.

## Acknowledgements
- The YOLO object detection model and its implementation are provided by [Ultralytics](https://ultralytics.com/).
- The Hugging Face transformers library is utilized to access the monocular depth estimation model, which is based on the work of [Depth Anything V2](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf).
- The Pepper robot integration is facilitated by the [NAOqi SDK](http://doc.aldebaran.com/2-4/naoqi/index.html) from SoftBank Robotics.
- I would also like to thank my supervisor Thomas Bolander for his guidance and support throughout this project.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
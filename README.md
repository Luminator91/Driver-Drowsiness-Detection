# Driver Drowsiness Detection

A real-time driver drowsiness detection system built using Python and OpenCV to help identify signs of driver fatigue and provide an audio warning.

## Features

* Real-time face detection using OpenCV's Deep Neural Network (DNN) module
* Eye detection using Haar cascades
* Audio alert when prolonged eye closure indicates possible drowsiness
* Webcam-based monitoring

## Technologies Used

* Python
* OpenCV
* NumPy
* Pygame
* Haar Cascade Classifier
* OpenCV DNN

## Project Structure

* `main.py` — Main detection and monitoring logic
* `deploy.prototxt` — Face detection model configuration
* `res10_300x300_ssd_iter_140000.caffemodel` — Pre-trained face detection model
* `alarm-danger-danger.mp3` — Audio alert

## How to Run

1. Install Python.

2. Install the required dependencies:

   ```bash
   pip install opencv-python numpy pygame
   ```

3. Ensure the model files and alarm audio are in the project directory.

4. Run the application:

   ```bash
   python main.py
   ```

5. Allow webcam access when prompted.

## Note

This is an educational computer vision project. It is not a certified driver-safety system and should not replace responsible driving practices.

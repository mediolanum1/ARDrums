# ARDrums: Real-Time Virtual Drumming

<p align="center">
  <a href="#about">About</a> •
  <a href="#installation">Installation</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#credits">Authors</a> •
  <a href="#license">License</a>
</p>


### About
Play a virtual 3D drum set using only a webcam and PC! This project uses the Mediapipe Pose Model along with anatomy-based 2D to 3D Pose Estimation to track your arms in real-time, allowing you to play a virtual drum set without any special hardware.


---

## Features
* **Minimal latency:** The system works in real-time with minimal latency that is unnoticeable to the human eye.
* **Zero Hardware Required:** Uses a standard webcam and computer vision to track motion.
* **True 3D Physics:** Custom Pythagorean depth estimation prevents tracking glitches and Z-fighting.
* **Kick Drum Support:** Tracks your right ankle to trigger the bass drum.

---

## Installation

Follow these steps to install the project:

**1. Clone the repository:**
```bash
git clone https://github.com/mediolanum1/ARDrums.git
cd ARDrums
```

**2.  Install all required packages
:**

   ```bash
   pip install -r requirements.txt
   ```


<br>


## 📥 Downloading Large Files (Git LFS)

This repository uses Git LFS (Large File Storage) to manage heavy model weights (e.g., MotionBERT checkpoints). A standard git clone may only download "pointer" files (tiny text files) instead of the actual binaries.

**1. Install Git LFS:**

Before cloning, ensure you have Git LFS installed on your machine.

    Windows: Download here or run winget install github.git-lfs

    macOS: brew install git-lfs

    Linux: sudo apt-get install git-lfs

Once installed, initialize it once on your system:
```Bash

git lfs install
```
**2. Clone the Repository**

Clone the repo as usual. Git LFS will automatically try to pull the large files during the clone process.
```Bash

git clone https://github.com/mediolanum1/ARDrums.git
```
**3. If files are missing (The "Pointer" Fix)**

If you already cloned the repo and your .task or .pth files in [pose_landmarker_models] folder are only a few bytes in size, run this command inside the project folder to fetch the real data:
```Bash

git lfs pull
```
   
---


## How To Use

Since this project uses Computer Vision, **your environment and your location in it matter**. Follow these 5 steps to get the best possible result:

   1. Ensure that the room you're playing in is well lit and you have enough space to stay at least ~1-2 meters from the camera
   2. Start the app. The webcam will start streaming and the system will start displaying video on the screen. Make sure that your shoulders, hips, and arms are visible.
   3. [OPTIONAL:] In case you want to play sitting and with bass kick, make sure that your left foot is fully visible
   4. After starting, the app will perform calibration. You will have 3 seconds to get into position. By the end of the calibration, you would need to stay still until the system displays the drum set.
   5. Play: swing your hands through the virtual drum set that is in front of you. You will see it mapped on the video output, as well as in the POV window in the top right corner. 


---

### Authors

:link: [Robert Li](https://github.com/mediolanum1)

:link: [Gunak Yuzak](https://github.com/GnkYzk)

### License

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](LICENSE)

# ARDrums

<br>

## Installation

Follow these steps to install the project:

1. Install all required packages

   ```bash
   pip install -r requirements.txt
   ```


<br>


## 📥 Downloading Large Files (Git LFS)

This repository uses Git LFS (Large File Storage) to manage heavy model weights (e.g., MotionBERT checkpoints). A standard git clone may only download "pointer" files (tiny text files) instead of the actual binaries.
1. Install Git LFS

Before cloning, ensure you have Git LFS installed on your machine.

    Windows: Download here or run winget install github.git-lfs

    macOS: brew install git-lfs

    Linux: sudo apt-get install git-lfs

Once installed, initialize it once on your system:
Bash

git lfs install

2. Clone the Repository

Clone the repo as usual. Git LFS will automatically try to pull the large files during the clone process.
Bash

git clone https://github.com/mediolanum1/ARDrums.git

3. If files are missing (The "Pointer" Fix)

If you already cloned the repo and your .bin or .pth files are only a few bytes in size, run this command inside the project folder to fetch the real data:
Bash

git lfs pull

4. Verify the Download

To confirm the models downloaded correctly, check the file size of the checkpoint:

    Expected size: ~150MB - 260MB

    Pointer size: < 1KB (If it's this small, run git lfs pull again)

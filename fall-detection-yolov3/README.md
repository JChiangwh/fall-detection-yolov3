Prerequisite technologies & frameworks
1) Python 3.5
2) PyTorch 1.0.1 download from https://pytorch.org/get-started/locally/
3) CUDA 9.0 or above download from https://developer.nvidia.com/cuda-10.0-download-archive
4) Official cfg and weights file download from https://pjreddie.com/darknet/yolo/
5) Anaconda environment (includes numpy, opencv) download from https://www.anaconda.com/
6) CUDNN
7) A good GPU preferably NVidia GTX 1080Ti
8) Ubuntu 16.04 operating system

Once everything from above is achieved you can:
1) Extract the zip file fall-detection-yolov3
2) Open command prompt
3) Type in "cd fall-detection-yolov3" to change the file directory
4) Once you're in the correct file directory type in "python fall.py" or "python fall.py --video [the input video's path]" to run the system.
5) The fall detector frame should then be presented on the screen upon entering the command mentioned above.
6) Hit the Q key to kill the program


******************************************************************NOTES****************************************************************************
1) if "error: CUDA out of memory" it means the GPU power in insufficient. 
2) The testing evidence is located in the output folder.


****************************************************************CREDITS****************************************************************************
@article{yolov3,
  title={YOLOv3: An Incremental Improvement},
  author={Redmon, Joseph and Farhadi, Ali},
  journal = {arXiv},
  year={2018}
}
http://leiluoray.com/2018/11/10/Implementing-YOLOV3-Using-PyTorch/
https://blog.paperspace.com/tag/series-yolo/

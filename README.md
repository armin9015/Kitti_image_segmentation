# Kitti_image_segmentation
Using KITTI dataset in order to perform image segmentation on an input directory, in this case "testing/image_2". The program iterates and processes every image found in said directory.
Code started from following this: https://pytorch.org/tutorials/beginner/deeplabv3_on_ios.html, to which I performed some code refactoring and added the following aspects: it now accepts a directory of input images instead of single images and it calculates the iot_score.

# How to run
1. Clone repo, replace the images in "testing/image_2" with your input images, delete images in output folder, delete and add your ground_truth images to the semantc_rgb folder.
2. Run main.py

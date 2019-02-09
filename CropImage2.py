# import the necessary packages
import argparse
import cv2
import numpy as np
import os
'''
Start the program.
Keep the images in ./images folder
Make a folder named './cropped'

'''

'''
Global Variables
'''
folder= './other/'
croppedFolder='./cropped_other/'
clone= []

# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not

ref_point = []
cropping = False

def shape_selection(event, x, y, flags, name):
  # grab references to the global variables
  global ref_point, cropping

  # if the left mouse button was clicked, record the starting
  # (x, y) coordinates and indicate that cropping is being
  # performed
  if event == cv2.EVENT_LBUTTONDOWN:
    ref_point = [(x, y)]
    cropping = True

  # check to see if the left mouse button was released
  elif event == cv2.EVENT_LBUTTONUP:
    # record the ending (x, y) coordinates and indicate that
    # the cropping operation is finished
    ref_point.append((x, y))
    cropping = False

    # draw a rectangle around the region of interest
 
    cv2.rectangle(image, ref_point[0], ref_point[1], (0, 255, 0), 4)
 
 
    
################# #Main Function begins here..###################################

for filename in os.listdir(folder):
    ref_point = []
    cropping = False
    image = cv2.imread(os.path.join(folder,filename)) 
 
    clone = image.copy()
    cv2.namedWindow(filename,cv2.WINDOW_NORMAL)
    cv2.resizeWindow(filename, 600, 400) 
    cv2.setMouseCallback(filename, shape_selection)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("n"):
      continue;
	
		
    # keep looping until the 'q' key is pressed
    while True:
      # display the image and wait for a keypress
      cv2.imshow(filename, image)
      key = cv2.waitKey(1) & 0xFF

      # if the 'r' key is pressed, reset the cropping region
      if key == ord("n"):
        print("skipping.. img.."+ str(filename)+"\n")
        break
      
      if key == ord("r"):
        image = clone.copy()
      # if the 'c' key is pressed, break from the loop
      elif key == ord("c"):
        break

		 
	 
    # if there are two reference points, then crop the region of interest
    # from teh image and display it
    if len(ref_point) == 2:
      crop_img = clone[ref_point[0][1]:ref_point[1][1], ref_point[0][0]:ref_point[1][0]] 
      cv2.namedWindow('crop_img',cv2.WINDOW_NORMAL)
      height = np.size(crop_img, 0)
      width = np.size(crop_img, 1)
      if(height>1  and width>1 ):
          cv2.resizeWindow('crop_img', 500*width/1000, 378*height/1000)
          cv2.imshow("crop_img", crop_img)
          print("cropping image ... and saving..")
          cv2.imwrite(croppedFolder + filename, crop_img)
          cv2.waitKey(0) 

    # close all open windows
    cv2.destroyAllWindows()

 


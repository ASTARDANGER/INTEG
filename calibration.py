## credits: https://nikatsanka.github.io/camera-calibration-using-opencv-and-python.html ##

import numpy as np
import cv2
import glob
import json
#import pathlib

def sort_corners(corners, nx, ny):
    # Trier les coins en fonction de leur position sur l'image
    sorted_corners = []
    for i in range(ny):
        # Trier par ligne : trier les coins de chaque ligne en fonction de leur coordonnée x
        row_corners = sorted(corners[i*nx:(i+1)*nx], key=lambda x: x[0][0])
        sorted_corners.extend(row_corners)
    
    # Convertir en array NumPy de type float32, qui est le format attendu
    return np.array(sorted_corners, dtype=np.float32)


# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (0.02,0,0), (0.04,0,0) ....,(0.12,0.12,0)
square_size = 0.02  # in mm
nx, ny = 7, 7 # number of corners in x and y directions
objp = np.zeros((nx*ny,3), np.float32)
objp[:,:2] = square_size*np.mgrid[0:nx,0:ny].T.reshape(-1,2)
print(objp)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob(r'photos_integ/img_et_pose_damier/*.jpg')

# path = 'results'
# pathlib.Path(path).mkdir(parents=True, exist_ok=True) 

found = 0
for fname in images:  # Here, 10 can be changed to whatever number you like to choose
    img = cv2.imread(fname) # Capture frame-by-frame
    #print(images[im_i])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (7,7), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)   # Certainly, every loop objp is the same, in 3D.
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        sorted_corners = sort_corners(corners2, nx, ny)
        imgpoints.append(sorted_corners)
        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (7,7), sorted_corners, ret)
        found += 1
        cv2.imshow('img', img)
        cv2.waitKey(500)
        # if you want to save images with detected corners 
        # uncomment following 2 lines and lines 5, 18 and 19
        # image_name = path + '/calibresult' + str(found) + '.png'
        # cv2.imwrite(image_name, img)

print("Number of images used for calibration: ", found)

# When everything done, release the capture
# cap.release()
cv2.destroyAllWindows()

# calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# transform the matrix and distortion coefficients to writable lists
data = {'camera_matrix': np.asarray(mtx).tolist(),
        'dist_coeff': np.asarray(dist).tolist(),
        'rvecs': np.asarray(rvecs).tolist(),
        'tvecs': np.asarray(tvecs).tolist()}

# and save it to a file
with open("calibration_matrix.json", "w") as f:
    json.dump(data, f)

# done
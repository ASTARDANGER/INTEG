### credits: https://nikatsanka.github.io/camera-calibration-using-opencv-and-python.html ###

import numpy as np
import cv2
import glob
import json
from math import atan2, pi
from natsort import natsorted

#import pathlib

###PARTIE 3###

def sort_corners(corners, nx, ny):
    # Trier les coins en fonction de leur position sur l'image
    sorted_corners = []
    for i in range(ny):
        # Trier par ligne : trier les coins de chaque ligne en fonction de leur coordonnée x
        row_corners = sorted(corners[i*nx:(i+1)*nx], key=lambda x: x[0][0])
        sorted_corners.extend(row_corners)
    
    # Convertir en array NumPy de type float32, qui est le format attendu
    return np.array(sorted_corners, dtype=np.float32)

def txt2matrix(filename):
    #"img_et_pose_damier/cart_poses.txt"
    #"pose_initiale/cart_poses.txt"
    with open(filename, "r", encoding="utf-8") as file: 
        next(file) # Ignorer la première ligne
        matrix = {}  # Dictionnaire pour stocker les matrices associées aux images
        for line in file:
            elements = line.strip().split(",")  # Séparer les éléments par ","

            keys = elements[0]  # Premier élément = nom de l'image

            values = elements[1:] # Deuxième élément = valeurs de la matrice
            values[0] = values[0].replace("(", "")  # Enlever "(" du premier nombre
            values[-1] = values[-1].replace(")", "")  # Enlever ")" du dernier nombre
            values = tuple(map(float, values))  # Deuxième élément = valeurs de la matrice
            values = np.array(values).reshape(4, 4)  # Transformer la liste en matrice 4x4
            values = values.T  # Transposer la matrice

            matrix[keys] = values # Stocker la matrice associée au nom de l'image
    return matrix

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0.02,0.02,0), (0.04,0.02,0), (0.06,0.02,0) ....,(0.14,0.14,0)
square_size = 0.02  # in mm
nx, ny = 7, 7 # number of corners in x and y directions
objp = np.zeros((nx*ny,3), np.float32)
objp[:,:2] = square_size*np.mgrid[1:nx+1,1:ny+1].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob(r'photos_integ/img_et_pose_damier/*.jpg')
images = natsorted(images)

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
        cv2.waitKey()
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
R = [cv2.Rodrigues(r)[0] for r in rvecs]
t = [e.tolist() for e in tvecs]

# transform the matrix and distortion coefficients to writable lists
data = {'camera_matrix': np.asarray(mtx).tolist(),
        'dist_coeff': np.asarray(dist).tolist(),
        'R': [r.tolist() for r in R],  # Convertir en listes
        'tvecs': t  # Si tvecs est aussi un ndarray
        }

# and save it to a file
with open("calibration_matrix.json", "w") as f:
    json.dump(data, f)

#affficher les paramètres ntrinsèques et extrinsèques
print("ret : "  ,ret)
print( "mtx : ", data['camera_matrix'] )
print( "dist_coeff : ", data['dist_coeff'])
print( "R : ", R[0] )
print("tvecs : ", data['tvecs'][0])
# Choisir la première image calibrée (index 0)
rvec = rvecs[0]
tvec = tvecs[0]
point_3D = np.array([objpoints[0][0]], dtype=np.float32)  # Premier point de la mire

# Projeter le point 3D sur l'image
point_2D, _ = cv2.projectPoints(point_3D, rvec, tvec, mtx, dist)

# Comparer avec le point détecté réel
point_reel = imgpoints[0][0]  # Premier point détecté dans l'image

print(f"Point projeté : {point_2D.ravel()}")
print(f"Point détecté réel : {point_reel.ravel()}")

# Calculer l'erreur de reprojection
erreur = np.linalg.norm(point_2D - point_reel)
print(f"Erreur de reprojection : {erreur:.2f} pixels")
# done

#afficher le point sur l'image
img=cv2.imread(images[0])

img = cv2.drawMarker(img, point_2D.astype(int).flatten(), (0,0,255),markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)
cv2.imshow('img',img)
cv2.waitKey()
#done

ax=2*atan2(640/2, mtx[0][0])
ay=2*atan2(480/2, mtx[1][1])
print("ax : ", ax*180/pi, "ay : " , ay*180/pi)


###PARTIE 4###

mat_poseInit = txt2matrix("photos_integ/pose_initiale/cart_poses.txt")
mat_poses = txt2matrix("photos_integ/img_et_pose_damier/cart_poses.txt")

def get_oTb(key):
    bTo = mat_poses[key]
    oTb = np.linalg.inv(bTo)
    return oTb

def get_mTc(i):
    cTm = np.zeros((4,4))
    cTm[:3,:3] = R[i]
    cTm[:3,3] = np.ravel(t[i]) #[[x],[y],[z]] -> [x,y,z]
    cTm[3,3] = 1
    return np.linalg.inv(cTm) #=mTc

def get_bTm():
    bTm = mat_poseInit["0"]
    return bTm

print(f"get_oTb: \n{get_oTb(str(0+1))}")
print(f"get_bTm: \n{get_bTm()}")
print(f"get_mTc: \n{get_mTc(0)}")


def get_oTc(key):
    return get_oTb(str(key+1)) @ get_bTm() @ get_mTc(key)

oTcList = []
for i in range(found):
    oTcList.append(get_oTc(i))
    oTcFinal = np.mean(oTcList,axis =0)
print( "OTC :",oTcFinal )
print("det(oTc)", np.linalg.det(oTcFinal[:3, :3]))
print("ortho : ", oTcFinal @ np.transpose(oTcFinal))
print("écart-type : ", np.std(oTcList, axis=0, ddof=0))





#vérification de l'erreur
def Erreur():
    for i in range(found):
        points_3D = np.array(objpoints[i]).squeeze()

        points_2D, _ = cv2.projectPoints(points_3D, rvecs[i], tvecs[i], mtx, dist)
        points_2D = point_2D.squeeze()

        points_reels  = np.array(imgpoints[i]).squeeze()

        erreurs = np.linalg.norm(points_2D - points_reels, axis =1)
        erreur_moy = np.mean(erreurs)

        print("Image", i+1)
        print("Erreur moyenne : ", erreur_moy)
        print("erreur max : ", max(erreurs))
        print("Erreur min : ", min(erreurs))


for i in  range(len(oTcList)):
    matctm = np.linalg.inv(get_oTb(str(i+1))@get_bTm())@oTcFinal
    matmtc = np.linalg.inv(matctm)
    tvec=np.array([matmtc[0][3],matmtc[1][3],matmtc[2][3]])

    rvec, _ =cv2.Rodrigues(matmtc[:3,:3])


    # Récupérer tous les points 3D du premier damier détecté
    points_3D = np.array(objpoints[i], dtype=np.float32)  # (1, 49, 3)


    # Projeter les points 3D en 2D
    points_2D, _ = cv2.projectPoints(points_3D, rvec, tvec, mtx, dist)  # (49, 1, 2)

    # Enlever les dimensions inutiles pour avoir (49, 2)
    points_2D = points_2D.squeeze()


    # Récupérer les points détectés correspondants (2D réels)
    points_reels = np.array(imgpoints[i], dtype=np.float32) 
    points_reels= np.squeeze(points_reels, axis= 1) # (49, 2)


    # Calculer l'erreur pour chaque point
    erreurs = np.linalg.norm(points_2D - points_reels, axis=1)  # Erreur pour chaque point
    # Calculer l'erreur moyenne
    erreur_moyenne = np.mean(erreurs)

    # Afficher les résultats
    print(f"Erreur de reprojection moyenne : {erreur_moyenne:.2f} pixels")
    print(f"Erreur maximale : {np.max(erreurs):.2f} pixels")
    print(f"Erreur minimale : {np.min(erreurs):.2f} pixels")

    img=cv2.imread(images[i])
    for points in points_2D:
        img = cv2.drawMarker(img, tuple(points.astype(int).flatten()), (0,0,255),markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)
    cv2.imshow('img',img)
    cv2.waitKey()
import numpy as np
import ast
import cv2

def load_camera_poses(filename):
    img_names = []
    poses = []
    with open(filename, 'r') as f:
        next(f)  # Skip first line
        lines = f.readlines()
        for line in lines:
            parts = line.split(",", 1)  # Split in 2 parts: the name of the image and the pose
            img_name = parts[0].strip()  # Get the image name part and remove leading/trailing whitespaces

            pose_str = parts[1].strip()  # Get the pose part and remove leading/trailing whitespaces
            try:
                img_names.append(img_name)

                # Remove parentheses and split by commas
                parse_pose = list(map(float, pose_str.strip("()").split(",")))
                # Reshape into a 4x4 matrix (column-major order)
                pose_matrix = np.array(parse_pose).reshape((4, 4), order='F')  # 'F' means column-major order   
                poses.append(pose_matrix)
            except (SyntaxError, ValueError):
                print(f"Parsing error line {line}")
    return img_names, poses

def generate_checkerboard_points(nx=8, ny=8, square_size=0.02):
    """
    Génère la liste des coordonnées (x, y, 0) de la mire dans son repère propre.
    nx : nombre de colonnes
    ny : nombre de lignes
    square_size : taille d'un carré de la mire en mètres (par défaut 2 cm)
    """
    object_points = []
    for i in range(ny):
        for j in range(nx):
            object_points.append((j * square_size, i * square_size, 0))  # z=0 as the sight is plane
    return np.array(object_points, dtype=np.float32)

def reorder_corners(corners, nx, ny):
    """
    Réorganise les coins détectés pour qu'ils suivent l'ordre attendu.
    corners : liste des coins détectés
    nx : nombre de colonnes
    ny : nombre de lignes
    """
    # transform into numpy array
    corners = np.squeeze(corners)  # (N, 1, 2) → (N, 2)

    # sort according to Y (line by line)
    corners = sorted(corners, key=lambda x: (x[1], x[0]))

    # We ensure each line is sorted from left to right
    reordered = []
    for i in range(ny):
        row = corners[i * nx:(i + 1) * nx]
        row = sorted(row, key=lambda x: x[0])  # Sorting from left to right
        reordered.extend(row)

    return np.array(reordered, dtype=np.float32)

if __name__ == "__main__":
    
    filename = "./photos_integ/img_et_pose_damier/cart_poses.txt"
    img_names = load_camera_poses(filename)[0]
    poses = load_camera_poses(filename)[1]
    # sight setting (here : 7x7 corners, 20 mm per square)
    nx, ny = 7, 7
    square_size = 0.02  # in mm
    obj_points = generate_checkerboard_points(nx, ny, square_size)
    print("Number of poses loaded :", len(poses))

    # Exemple de détection avec OpenCV
    for img_name in img_names:
        image = cv2.imread(f"./photos_integ/img_et_pose_damier/{img_name}.jpg")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        found, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        if found:
            ordered_corners = reorder_corners(corners, nx, ny)
        else:
            print("Sight undetected")
    
    # Manufacturer's intrinsic parameters (in pixels)
    cx, cy = 324.9, 245.8
    fx, fy = 607.9, 607.8
    img_width, img_height = 640, 480

    # Compute field of view (FoV)
    fov_x = 2 * np.arctan(img_width / (2 * fx)) * 180 / np.pi
    fov_y = 2 * np.arctan(img_height / (2 * fy)) * 180 / np.pi

    print(f"Computed FoV: Horizontal = {fov_x:.2f}°, Vertical = {fov_y:.2f}°")
    print("Expected FoV: 55.52° (horizontal), 43.09° (vertical)")

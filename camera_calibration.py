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
                pose = ast.literal_eval(pose_str)  # Convert in tuple of floats
                poses.append(pose)
            except (SyntaxError, ValueError):
                print(f"Parsing error line {line}")
    return img_names, poses

def generate_chessboard_points(board_size, square_size):
    objp = np.zeros((board_size * board_size, 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size, 0:board_size].T.reshape(-1, 2)
    objp *= square_size
    return objp

if __name__ == "__main__":
    filename = "./photos_integ/img_et_pose_damier/cart_poses.txt"
    img_names = load_camera_poses(filename)[0]
    poses = load_camera_poses(filename)[1]
    # sight setting (here : 8x8 squares, 20 mm per square)
    board_size = 8
    square_size = 20.0  # in mm
    obj_points = generate_chessboard_points(board_size, square_size)
    print("Number of poses loaded :", len(poses))

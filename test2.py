import cv2
import numpy as np
import glob
import os

# Dossier contenant les images
image_folder = "photos_integ/flux_continu"
output_folder = "output_images"
os.makedirs(output_folder, exist_ok=True)

def process_image(image_path):
    img = cv2.imread(image_path)

    # Convertir en HSV et extraire la valeur
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]

    # Égalisation d'histogramme pour améliorer le contraste
    v_equalized = cv2.equalizeHist(v_channel)

    # Seuillage d'Otsu pour détecter les blobs noirs
    _, thresh = cv2.threshold(v_equalized, 40, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Opérations morphologiques (dilatation puis érosion) pour améliorer la segmentation
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Détection des contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sélection du plus grand contour en surface
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)

        # Calcul du centroïde
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.circle(img, (cX, cY), 5, (0, 255, 0), -1)  # Centroïde en vert
            cv2.putText(img, f"({cX}, {cY})", (cX - 40, cY - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Dessin du contour
        cv2.drawContours(img, [largest_contour], -1, (0, 0, 255), 2)  # Contour en rouge

    return img

# Traiter toutes les images du dossier
images = glob.glob(os.path.join(image_folder, "*.jpg"))

for image_path in images:
    processed_img = process_image(image_path)
    
    
    # Afficher l'image (décommentez pour voir chaque image)
    cv2.imshow("Processed Image", processed_img)
    cv2.waitKey(0)

cv2.destroyAllWindows()

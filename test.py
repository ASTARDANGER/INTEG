import cv2
import numpy as np
import glob

# Fonction pour ajuster un cercle tangent au contour externe
def fit_tangent_circle(contour):
    # Convertir le contour en un tableau de points
    points = contour.reshape(-1, 2)
    
    # Trouver le centre du contour (centre des moments)
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = 0, 0

    # Calculer la distance de chaque point du contour au centre
    distances = np.linalg.norm(points - np.array([cx, cy]), axis=1)
    
    # Trouver la distance maximale (cercle tangent)
    radius = int(np.max(distances))  # Rayon du cercle = distance maximale

    return (cx, cy), radius

# Charger l'image
images = glob.glob(r'photos_integ/flux_continu/*.jpg')

for image_path in images:
    img = cv2.imread(image_path)

    def detect_black_hole(img):
        # Convertir en niveaux de gris
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Appliquer un seuillage pour isoler les zones sombres
        _, thresh = cv2.threshold(gray, 18, 255, cv2.THRESH_BINARY_INV)
        
        # Trouver les contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filtrer les contours selon leur taille et leur circularité
        best_contour = None
        best_circularity = 0
        for c in contours:
            area = cv2.contourArea(c)
            perimeter = cv2.arcLength(c, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * (area / (perimeter ** 2))  # 1 pour un cercle parfait
            if area > 600 and circularity > best_circularity:  # Ajuster 500 selon la taille du trou si besoin de filtrer petits cercles parasites
                best_circularity = circularity
                best_contour = c
        
        if best_contour is not None:
            # Ajuster un cercle tangent au contour externe
            (cx, cy), radius = fit_tangent_circle(best_contour)
            (cx, cy) = (int(cx), int(cy))
            # Dessiner le cercle ajusté
            cv2.circle(img, (cx, cy), radius, (0, 0, 255), 2)  # Rouge pour le cercle
            
            # Dessiner le centre du cercle
            cv2.circle(img, (cx, cy), 7, (0, 255, 0), -1)  # Vert pour marquer le centre du cercle
            cv2.putText(img, f"({cx}, {cy})", (cx - 30, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, (0, 255, 0), 2)
        
        return img

    # Appliquer la détection
    result = detect_black_hole(img)

    # Afficher l'image résultante
    cv2.imshow("Detected Hole", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

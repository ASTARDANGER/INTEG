import cv2
import numpy as np
import glob

# Fonction pour ajuster un cercle tangent au contour
def fit_circle_to_contour(contour):
    # Convertir le contour en un tableau de points
    points = contour.reshape(-1, 2)
    
    # Calculer le centre et le rayon du cercle via une méthode des moindres carrés
    A = np.c_[2 * points[:, 0], 2 * points[:, 1], np.ones(points.shape[0])]
    b = points[:, 0] ** 2 + points[:, 1] ** 2
    params = np.linalg.lstsq(A, b, rcond=None)[0]
    
    # Extraire le centre (cx, cy) et le rayon
    cx = params[0]
    cy = params[1]
    r = np.sqrt(params[2] + cx**2 + cy**2)
    
    return (int(cx), int(cy)), int(r)

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
            # Calcul du centroïde
            M = cv2.moments(best_contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = 0, 0
            
            # Dessiner le contour
            cv2.drawContours(img, [best_contour], -1, (0, 0, 255), 2)
            # Dessiner le centroïde
            cv2.circle(img, (cX, cY), 7, (0, 255, 0), -1)  # Rouge pour marquer le trou
            cv2.putText(img, f"({cX}, {cY})", (cX - 30, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, (0, 255, 0), 2)
            
            # Ajuster un cercle qui passe par le plus de points du contour
            (cx, cy), radius = fit_circle_to_contour(best_contour)
            
            # Dessiner le cercle ajusté
            cv2.circle(img, (cx, cy), radius, (255, 0, 0), 2)  # Rouge pour le cercle
            
            # Dessiner le centre du cercle
            cv2.circle(img, (cx, cy), 7, (255, 0, 0), -1)  # Vert pour marquer le centre du cercle
            cv2.putText(img, f"({cx}, {cy})", (cx - 30, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, (255, 0, 0), 2)
        
        return img

    # Appliquer la détection
    result = detect_black_hole(img)

    # Afficher l'image résultante
    cv2.imshow("Detected Hole", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
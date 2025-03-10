import cv2
import numpy as np
import glob

# Charge l'image
images = glob.glob(r'photos_integ/flux_continu/*.jpg')
u, v, a = 0, 0, 0

# Fonction pour ajuster un cercle tangent au contour
def fit_circle_to_contour(contour):
    # Conversion du contour en un tableau de points
    points = contour.reshape(-1, 2)
    
    # Calcul du centre et du rayon du cercle via une méthode des moindres carrés
    A = np.c_[2 * points[:, 0], 2 * points[:, 1], np.ones(points.shape[0])]
    b = points[:, 0] ** 2 + points[:, 1] ** 2
    params = np.linalg.lstsq(A, b, rcond=None)[0]
    
    # Extraction du centre (cx, cy) et du rayon
    cx = params[0]
    cy = params[1]
    r = np.sqrt(params[2] + cx**2 + cy**2)
    
    return (int(cx), int(cy)), int(r)

def detect_black_hole(img):
    # Conversion en niveaux de gris
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # On applique un seuillage pour isoler les zones sombres
    _, thresh = cv2.threshold(gray, 18, 255, cv2.THRESH_BINARY_INV)
    
    # On cherche les contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filtrage des contours selon leur taille et leur circularité
    best_contour = max(contours, key=cv2.contourArea, default=None) # Contour le plus grand
    
    if best_contour is not None:
        # Calcul du centroïde du contour détecté
        M = cv2.moments(best_contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0
        
        # Dessin du contour
        cv2.drawContours(img, [best_contour], -1, (0, 0, 255), 2) # Rouge pour le contour
        # Dessin du centroïde
        cv2.circle(img, (cX, cY), 7, (0, 255, 0), -1)  # Vert pour le centroïde
        cv2.putText(img, f"({cX}, {cY})", (cX - 30, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, (0, 255, 0), 2)
        
        # Ajustement pour avoir un cercle qui épouse le mieux le contour
        (cx, cy), radius = fit_circle_to_contour(best_contour)
        
        # Dessin du cercle ajusté
        cv2.circle(img, (cx, cy), radius, (255, 0, 0), 2) # Bleu pour le cercle ajusté
        
        # Dessin du centre du cercle
        cv2.circle(img, (cx, cy), 7, (255, 0, 0), -1)  # Bleu pour marquer le centre du cercle
        cv2.putText(img, f"({cx}, {cy})", (cx - 30, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
        return img, best_contour, cx, cy, radius
    return None, None, None, None

for image_path in images:
    img = cv2.imread(image_path)

    ## ON ROGNE L'IMAGE POUR ISOLER LE TROU
    # Conversion en HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Définition des seuils pour la détection du bleu
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([130, 255, 255])

    # Création d'un masque
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # On cherche les contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # On cherche le plus grand contour
        largest_contour = max(contours, key=cv2.contourArea)

        # Obtention de la boîte englobante
        x, y, w, h = cv2.boundingRect(largest_contour)

        # On Rogne l'image
        cropped_image = img[y:y+h, x:x+w].copy()
    ## FIN DU ROGNAGE

    # On applique la détection
    result_img, best_contour, cx, cy, radius = detect_black_hole(cropped_image)
    updated_contour = best_contour + np.array([x, y]) # Mise à jour des coordonnées du contour dans le référentiel de l'image d'origine
    if result_img is None:
        continue
    ## REMAP LES RÉSULTATS SUR L'IMAGE ORIGINALE
    # Conversion des coordonnées pour repasser dans le référentiel de l'image d'origine
    cx += x
    cy += y
    # Dessin des résultats sur l'image d'origine
    cv2.drawContours(img, updated_contour, -1, (0, 0, 255), 2) # Rouge pour le contour
    cv2.circle(img, (cx, cy), radius, (255, 0, 0), 2)  # Cercle bleu
    cv2.circle(img, (cx, cy), 7, (0, 255, 0), -1)  # Centre vert
    cv2.putText(img, f"({cx}, {cy})", (cx - 30, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, (0, 255, 0), 2)
    ## FIN DU REMAP     
    print(cx, cy, np.pi*radius**2)
    # Affichage de l'image
    cv2.imshow("Detected Hole", img) #mettre img pour visualiser dans l'image d'origine et cropped_image pour visualiser dans l'image rognée
    
    # Nous avons mesuré l'aire du trou sur l'image où l'organe terminal est suffisamment proche de celui-ci (en pixels)
    a_ref = np.pi*37**2 # Aire du trou en pixels
    epsilon = 5 # Marge d'erreur tolérée (en pixels) <=> à quel point le trou est proche de l'organe terminal
    if abs(np.pi*radius**2-a_ref) < epsilon:
        u = cx
        v = cy
        a = np.pi*radius**2
        print("got it")
        break
    cv2.waitKey(300)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(u, v, a)
#!/usr/bin/env python
# IL Y PEUT DES TABs A LA PLACE DES ESPACES PARFOIS
import rospy
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import WrenchStamped, TwistStamped
from franka_msgs.msg import FrankaState
from cv_bridge import CvBridge
import cv2
import numpy as np

from pynput import keyboard

def wait_for_keypress(target_key='i'):
    def on_press(key):
        try:
            if key.char == target_key:  # Vérifie si la touche pressée est 'i'
                return False  # Arrête le listener
        except AttributeError:
            pass  # Ignore les touches spéciales

    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()  # Attend que la touche soit pressée


class IntegVisualServoingNode:
    def __init__(self):
        # Initialize the node
        rospy.init_node('integ_visual_servoing_node', anonymous=True)

        # Subscribers
        rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)
        rospy.Subscriber('/franka_state_controller/joint_states', JointState, self.joint_states_callback)
        rospy.Subscriber('/franka_state_controller/franka_states', FrankaState, self.cart_states_callback)
        rospy.Subscriber('/franka_state_controller/F_ext', WrenchStamped, self.f_ext_callback)
        # Publisher
        self.cart_vel_pub = rospy.Publisher('/cart_vel_desired_', TwistStamped, queue_size=10)

        # Rate of the loop
        self.rate = rospy.Rate(10)

        self.bridge = CvBridge()
        self.latest_image = None
        self.current_joint_positions = None
        self.current_cartesian_pose = None
        self.current_force = None

        ### AJOUT
        # Seuil de force pour arrêter l'insertion
        self.force_threshold = 5.0  # en Newtons
        # Valeur de s_star (la position de notre trou idéal)
        self.s_star = np.array([369, 100])  # À COMPLETER #402 110 np.array([369, 112])
        self.s_a_star = 6600 #(6600) 4300
        # Valeur de s-dot pour savoir quand est-ce qu'on est sur la cible
        self.s_dot = None
        self.s_a_dot = None
        self.trigger = False
        ### AJOUT

    def joint_states_callback(self, msg):
        self.current_joint_positions = msg.position  # Tuple of joint positions

    def cart_states_callback(self, msg):
        # Store the current Cartesian pose of the robot
        pose = msg.O_T_EE
        self.current_cartesian_pose = (
            pose[0], pose[1], pose[2], pose[3],
            pose[4], pose[5], pose[6], pose[7],
            pose[8], pose[9], pose[10], pose[11],
            pose[12], pose[13], pose[14], pose[15]
        )

    def f_ext_callback(self, msg):
        self.current_force = msg.wrench

    def image_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image and store it
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            cv2.circle(self.latest_image, (369,100), radius=2, color=(0, 0, 255), thickness=-1)
            cv2.imshow("Iaaaaa", self.latest_image)
            cv2.waitKey(1)  # Ajoute un petit délai pour afficher l'image
        except Exception as e:
            rospy.logerr(f"Failed to process image: {e}")



    ###AJOUT
    def detect_hole(self, img):
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
            _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

            # On cherche les contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            #for point in contours:
            #    cv2.circle(gray, (point[0][0][0] , point[0][0][1]) , radius=2, color=(0, 0, 255), thickness=-1)
            #cv2.imshow("contours", gray)
            #cv2.waitKey(1)
            # Filtrage des contours selon leur taille et leur circularité
            best_contour = max(contours, key=cv2.contourArea, default=None)  # Contour le plus grand

            if best_contour is not None:
                # Ajustement pour avoir un cercle qui épouse le mieux le contour

                (cx, cy), radius = fit_circle_to_contour(best_contour)
                cv2.circle(img, (cx, cy) , radius, (255,0,0))
                cv2.imshow("Image reçue", img)
                return cx, cy, radius
            return 369, 100, 7000

        u, v, a = 369, 100, 8000
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
            cropping_plus = -int( min(h,w)//5)#pour rogner encore plus l'image
            # On Rogne l'image
            cropped_image = img[y-int(cropping_plus*0.5):y+h+cropping_plus, x-cropping_plus:x+w+int(cropping_plus*1.6)].copy()
        ## FIN DU ROGNAGE

        # On applique la détection
        cx, cy, radius = detect_black_hole(cropped_image)

        if cx is None:
            rospy.loginfo()
            return u, v, a
        ## REMAP LES RÉSULTATS SUR L'IMAGE ORIGINALE
        # Conversion des coordonnées pour repasser dans le référentiel de l'image d'origine
        cx += x -cropping_plus
        cy += y -int(cropping_plus*0.5)
        ## FIN DU REMAP
        #print(cx, cy, np.pi * radius**2)
        # Nous avons mesuré l'aire du trou sur l'image où l'organe terminal est suffisamment proche de celui-ci (en pixels)
        a_ref = np.pi * 37**2  # Aire du trou en pixels
        epsilon = 5  # Marge d'erreur tolérée (en pixels) <=> à quel point le trou est proche de l'organe terminal

        #if abs(np.pi * radius**2 - a_ref) > epsilon:
        u = cx
        v = cy
        a = np.pi * radius**2

        return u, v, a
    ###AJOUT







    ### AJOUT
    # Nouvelle idée de loi de commande (courtoisie de Jerem) :
    # On prend les coordonnées du cercle sur la dernière image où le cercle
    # n'est pas en intersection avec l'organe.
    # Tant que le centre de ce cercle et celui sur l'image actuelle ne sont pas alignés,
    # on fait la différence entre les deux pour choisir V_x et V_y.
    # On prend aussi la surface du cercle pour cette dernière image. Tant que celle
    # de l'image actuelle n'est pas la même, on fait la différence pour choisir V_z.
    # Il est plus sage de d'abord faire la première partie, puis ensuite faire un mix des deux autres parties.
    def compute_velocity(self):
        """
        Calcule la vitesse cartésienne désirée en fonction de la position du trou dans l'image.
        """
        if self.latest_image is None:
            return None

        # Détection du trou (return u, v, a)
        result = self.detect_hole(self.latest_image)
        if result is None:
            return None
        u, v, area = result
        s = np.array([u, v, 0])

        # Paramètres de la caméra (centre de l'image + focale)
        u0, v0, f_u, f_v = 320, 240, 607.9, 607.8

        x, y, Z = (u - u0) / f_u, (v - v0) / f_v, 1
        lambd = 0.0003

        # Calcul de Ls (À vérifier)
        L_s = np.array([
            [-1 / Z, 0, x / Z],
            [0, -1 / Z, y / Z]
        ])

        # Inversion pseudo-généralisée de Ls
        L_s_inv = np.linalg.pinv(L_s)

        # Calcul de s_dot
        self.s_dot = np.array([s[0] - self.s_star[0], s[1] - self.s_star[1]])

        # Obtenir bRc
        oTc = np.array([[-0.016, 0.999, 0.017, 0.045],
                        [-0.999, -0.016, 0.018, 0.029],
                        [0.018, -0.017, 0.999, -0.134],
                        [0, 0, 0, 1]])
        bTo = np.array([[self.current_cartesian_pose[0], self.current_cartesian_pose[4], self.current_cartesian_pose[8], self.current_cartesian_pose[12]],
                        [self.current_cartesian_pose[1], self.current_cartesian_pose[5], self.current_cartesian_pose[9], self.current_cartesian_pose[13]],
                        [self.current_cartesian_pose[2], self.current_cartesian_pose[6], self.current_cartesian_pose[10], self.current_cartesian_pose[14]],
                        [self.current_cartesian_pose[3], self.current_cartesian_pose[7], self.current_cartesian_pose[11], self.current_cartesian_pose[15]]])
        bRc = (bTo @ oTc)[:3, :3]

        # Calculer v_plan
        v_plan = -lambd * bRc @ L_s_inv @ self.s_dot

        # Ajouter une consigne de vitesse en Z
        self.s_a_dot = self.s_a_star - area
        v_z = - min( 0.001**2 * self.s_a_dot , 0.01)

        # Envoyer bVo en consigne
        return np.array([v_plan[0], v_plan[1], v_z])
    ### AJOUT







    def run(self):
        while not rospy.is_shutdown():
            twist_msg = TwistStamped()
            twist_msg.header.stamp = rospy.Time.now()

            # AJOUT
            velocity = self.compute_velocity()
            print(self.s_a_dot)

            if velocity is not None and abs(self.s_a_dot) >300 and self.trigger != True:
                """
                rospy.loginfo("Aligné, descente en cours...")
                twist_msg.twist.linear.x = velocity[0]
                twist_msg.twist.linear.y = velocity[1]
                twist_msg.twist.linear.z = velocity[2]
                """

                # Déclenchement de l'insertion en Z si aligné
                if abs(velocity[0]) < 0.001 and abs(velocity[1]) < 0.001 :
                    rospy.loginfo("Aligné, descente en cours...")
                    twist_msg.twist.linear.z = velocity[2]
                    twist_msg.twist.linear.x = 0.0
                    twist_msg.twist.linear.y = 0.0
                else :
                    twist_msg.twist.linear.x = velocity[0]
                    twist_msg.twist.linear.y = velocity[1]
                    twist_msg.twist.linear.z = 0.0


            elif self.s_a_dot ==None :

                twist_msg.twist.linear.x = 0.0
                twist_msg.twist.linear.y = 0.0
                twist_msg.twist.linear.z = 0.0
            else :
                if self.trigger == True :

                    rospy.loginfo("Aligné, insertion en cours...")
                    twist_msg.twist.linear.x = +0.0002
                    twist_msg.twist.linear.y = 0.0004
                    twist_msg.twist.linear.z = -0.004
                else :
                    twist_msg.twist.linear.x = 0.0
                    twist_msg.twist.linear.y = 0.0
                    twist_msg.twist.linear.z = 0.0
                    rospy.logwarn("Press i")
                    wait_for_keypress()
                    self.trigger = True


            # Vérification de la force de contact
            if self.current_force is not None and abs(self.current_force.force.z) > self.force_threshold:
                rospy.logwarn("Force seuil atteinte, arrêt du mouvement.")
                twist_msg.twist.linear.x = 0.0
                twist_msg.twist.linear.y = 0.0
                twist_msg.twist.linear.z = 0.0
            # AJOUT
            twist_msg.twist.angular.x = 0.0
            twist_msg.twist.angular.y = 0.0
            twist_msg.twist.angular.z = 0.0

            self.cart_vel_pub.publish(twist_msg)
            self.rate.sleep()

if __name__ == '__main__':
    try:
        node = IntegVisualServoingNode()
        node.run()
    except rospy.ROSInterruptException:
        pass

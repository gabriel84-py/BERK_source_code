import cv2
import cv2.aruco as aruco

# choisir dictionnaire
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

# générer un marqueur
marker_id = 2
size = 400  # pixels

marker = aruco.generateImageMarker(dictionary, marker_id, size)

cv2.imwrite("aruco_0.png", marker)
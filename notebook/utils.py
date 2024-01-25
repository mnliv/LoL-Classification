import numpy as np
import cv2


def detect_circles(img):
    # image = cv2.imread(image_path)
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # img_blur = cv2.medianBlur(img, 5)

    circles = cv2.HoughCircles(
        img,     
        cv2.HOUGH_GRADIENT, 
        dp=1,           # Inverse ratio of the accumulator resolution to the image resolution (1 means same resolution)
        minDist=30,     # Minimum distance between the centers of the detected circles
        param1=50,      # Gradient value used in the edge detection
        param2=60,      # Accumulator threshold for circle detection (lower value means more circles)
        minRadius=0,    # Minimum radius of the detected circles
        maxRadius=0
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        largest_circle = max(circles[0, :], key=lambda x: x[2])
        x, y, radius = largest_circle[0], largest_circle[1], largest_circle[2]
        cropped_image = img[y - radius:y + radius, x - radius:x + radius]

        return True, cropped_image
    return False, None

def detect_rectangles(img):
    # image = cv2.imread(image_path)
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [contour for contour in contours if cv2.contourArea(contour) > 10]
    
    largest_rectangle = None
    max_area = 0

    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 4:  # Assuming rectangles
            area = cv2.contourArea(approx)
            if area > max_area:
                max_area = area
                largest_rectangle = approx
    if largest_rectangle is not None:
        x, y, w, h = cv2.boundingRect(largest_rectangle)
        cropped_image = img[y:y+h, x:x+w]
        return True, cropped_image
    return False, None
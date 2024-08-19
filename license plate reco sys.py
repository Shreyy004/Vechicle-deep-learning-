import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"

# Read the image file
image = cv2.imread("car1.jpg")
cv2.imshow("Original", image)
cv2.waitKey(0)

# Convert to grayscale image
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray Image", gray_image)
cv2.waitKey(0)

# Canny edge detection
canny_edge = cv2.Canny(gray_image, 170, 200)
cv2.imshow("Canny Edge", canny_edge)
cv2.waitKey(0)

# Find the contours based on edges
contours, _ = cv2.findContours(canny_edge.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]

# Initialize license plate contour and x, y, w, h coordinates
contour_with_license_plate = None
license_plate = None
x = None
y = None
w = None
h = None

# Create a blank canvas to draw contours on
contour_image = image.copy()

# Draw contours on the blank canvas
cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

# Display the image with contours and the drawn license plate
cv2.imshow("Image with Contour and License Plate", contour_image)
cv2.waitKey(0)

# Find the contour with 4 potential corners and create ROI around it
for contour in contours:
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
    if len(approx) == 4:
        contour_with_license_plate = approx
        x, y, w, h = cv2.boundingRect(contour)
        license_plate = gray_image[y:y + h, x:x + w]
        break

if license_plate is not None:
    (thresh, license_plate) = cv2.threshold(license_plate, 127, 255, cv2.THRESH_BINARY)
    cv2.imshow("License Plate", license_plate)
    cv2.waitKey(0)

    # Removing noise from the detected image before sending it to Tesseract
    license_plate = cv2.bilateralFilter(license_plate, 11, 17, 17)

    # Text recognition
    text = pytesseract.image_to_string(license_plate)

    image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 3)
    image = cv2.putText(image, text, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Detected License Plate", image)
    cv2.waitKey(0)
    print("License Plate:", text)

cv2.destroyAllWindows()

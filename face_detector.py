from keras.models import load_model
import cv2
import numpy as np
import time

class FaceDetector:
	def __init__(self):
		pass

	def get_face_and_predict(self):

		face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

		cap = cv2.VideoCapture(0)
		for i in range(30):
		    _, img = cap.read()
		gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		# plt.imshow(gray_image)
		faces_rects = face_cascade.detectMultiScale(gray_image, 1.1, 4)
		# cv2.waitKey(150)
		for (x, y, w, h) in faces_rects:
		    cv2.rectangle(gray_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
		    img = gray_image[y:y + h, x:x + w]
		    # plt.imshow(img)
		img = cv2.resize(src=img, dsize=(48,48))
		img = np.expand_dims(np.expand_dims(img, 0), 3)
		cap.release()
		model = load_model('model_faces.h5')
		start = time.time()
		prediction = model.predict(img)
		end = time.time()
		print(f"PREDICTIONTIME: {end-start}")
		return prediction[0][0]
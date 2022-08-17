import numpy as np
import cv2
from datetime import datetime


def zoom_at(img, zoom, coord=None):
  """
  Simple image zooming without boundary checking.
  Centered at "coord", if given, else the image center.

  img: numpy.ndarray of shape (h,w,:)
  zoom: float
  coord: (float, float)
  """
  # Translate to zoomed coordinates
  h, w, _ = [zoom * i for i in img.shape]

  if coord is None:
    cx, cy = w / 2, h / 2
  else:
    cx, cy = [zoom * c for c in coord]

  img = cv2.resize(img, (0, 0), fx=zoom, fy=zoom)
  img = img[int(round(cy - h / zoom * .5)): int(round(cy + h / zoom * .5)),
        int(round(cx - w / zoom * .5)): int(round(cx + w / zoom * .5)),
        :]

  return img


def Detectotron():

  previous_frame = None
  cap = cv2.VideoCapture(0)
  frame_width = int(cap.get(3))
  frame_height = int(cap.get(4))

  out = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 5, (frame_width, frame_height))
  out2 = cv2.VideoWriter('outpy-2.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 5, (frame_width, frame_height))
  # out3 = cv2.VideoWriter('outpy-3.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 5, (1000, 1000))
  while True:
    ret, img_brg = cap.read()

    img_brg = cv2.cvtColor(src=img_brg, code=cv2.COLOR_BGR2RGB)

    # 1. Load image; convert to RGB
    img_brg = np.array(img_brg)
    img_rgb = cv2.cvtColor(src=img_brg, code=cv2.COLOR_BGR2RGB)
    img_ogn = cv2.cvtColor(src=img_brg, code=cv2.COLOR_BGR2RGB)
    # 2. Prepare image; grayscale and blur
    prepared_frame = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    prepared_frame = cv2.GaussianBlur(src=prepared_frame, ksize=(5, 5), sigmaX=0)

    # 2. Calculate the difference
    if (previous_frame is None):
      # First frame; there is no previous one yet
      previous_frame = prepared_frame
      continue

    # 3. Set previous frame and continue if there is None
    if (previous_frame is None):
      # First frame; there is no previous one yet
      previous_frame = prepared_frame
      continue

    # calculate difference and update previous frame
    diff_frame = cv2.absdiff(src1=previous_frame, src2=prepared_frame)
    previous_frame = prepared_frame

    # 4. Dilute the image a bit to make differences more seeable; more suitable for contour detection
    kernel = np.ones((5, 5))
    diff_frame = cv2.dilate(diff_frame, kernel, 1)

    # 5. Only take different areas that are dif ferent enough (>20 / 255)
    thresh_frame = cv2.threshold(src=diff_frame, thresh=25, maxval=255, type=cv2.THRESH_BINARY)[1]

    # 6. Find and optionally draw contours
    contours, _ = cv2.findContours(image=thresh_frame, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

    # Comment below to stop drawing contours
    cv2.drawContours(image=img_rgb, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA)

    # Uncomment 6 lines below to stop drawing rectangles
    cv2.imshow('DETECTOTRON 1.0', img_rgb)
    v = 0
    for contour in contours:


     if cv2.contourArea(contour) < 5:
      # too small: skip!
      v = 0
      continue

     elif cv2.contourArea(contour) >= 5:
      v = 1
    if (cv2.waitKey(30) == 27):
      # out.release()
        break

    x = 0
    y = 0
    if v >= 1:

        (x, y, w, h) = cv2.boundingRect(contour)
        print (x, y, w * h)

        # Text on frames
        font = cv2.FONT_HERSHEY_PLAIN
        cv2.putText(img_rgb, str(datetime.now()), (20, 40),
                  font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        hahaha = x, y, w * h
        hahaha = str(hahaha)
        cv2.putText(img_rgb, hahaha, (40, 60),
                  font, 1, (255, 255, 255), 2, cv2.LINE_AA)

        #Output video

        out.write(img_rgb)
        out2.write(img_ogn)
        # img4 = zoom_at(img_rgb, 4, coord=(x, y))
        # out3.write(img4)

  cap.release()
  out.release()
  out2.release()
  out3.release()

  # Cleanup
  cv2.destroyAllWindows()



Detectotron()

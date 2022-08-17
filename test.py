from tkinter import *
import cv2
from PIL import ImageTk, Image
import numpy as np
from datetime import datetime
def video_stream():
    cv2image = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGBA)
    cv2image2 = cv2.cvtColor(img_ogn, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    img2 = Image.fromarray(cv2image2)
    imgtk = ImageTk.PhotoImage(image=img)
    imgtk2 = ImageTk.PhotoImage(image=img2)
    lmain.imgtk = imgtk
    lmain.imgtk = imgtk2
    lmain.configure(image=imgtk)
    lmain.configure(image=imgtk2)
    lmain2.imgtk = imgtk
    lmain2.imgtk = imgtk2
    lmain2.configure(image=imgtk)
    lmain2.configure(image=imgtk2)

root = Tk()
root.title("Detectotron 1.0")
root.geometry('2500x1000')
root.resizable(0, 0)
cap = cv2.VideoCapture(0)
label =Label(root)
label.grid(row=0, column=2)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out1 = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 5, (frame_width, frame_height))
out2 = cv2.VideoWriter('outpy-2.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 5, (frame_width, frame_height))

Live_frame = Frame(root, bg='black', width = 200, height=200).grid(column=0, row=0, sticky=(N, W, S ))
Det_frame = Frame(root, bg='black', width = 200, height=200).grid(column=1, row=0, sticky=(N, E, S ))

lmain = Label(Live_frame)
lmain.grid(column=0, row=0)
lmain2 = Label(Det_frame)
lmain2.grid(column=1, row=0)






# ret, Live_frame = cap.read()
# ret, Det_frame = cap.read()
previous_frame = None

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
    cv2.drawContours(image=img_rgb, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=1,
                     lineType=cv2.LINE_AA)

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
        print(x, y, w * h)

        # Text on frames
        font = cv2.FONT_HERSHEY_PLAIN
        cv2.putText(img_rgb, str(datetime.now()), (20, 40),
                    font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        hahaha = x, y, w * h
        hahaha = str(hahaha)
        cv2.putText(img_rgb, hahaha, (40, 60),
                    font, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Output video
        out1.write(img_rgb)
        out2.write(img_ogn)
        # img4 = zoom_at(img_rgb, 4, coord=(x, y))
        # out3.write(img4)

    ret, Live_frame = cap.read()
    ret, Det_frame = cap.read()
    cv2.imshow("Live Feed", Live_frame)
    cv2.imshow("Det Feed", Det_frame)
    cv2image = cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2RGB)
    img = Image.fromarray(cv2image)
    # Convert image to PhotoImage
    imgtk = ImageTk.PhotoImage(image=img)
    label.imgtk = imgtk
    label.configure(image=imgtk)
    # Repeat after an interval to capture continiously
    # label.after(20, show_frames)
    if cv2.waitKey(1) & 0xFF == 27:
        break
    cv2image = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGBA)
    cv2image2 = cv2.cvtColor(img_ogn, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    img2 = Image.fromarray(cv2image2)
    imgtk = ImageTk.PhotoImage(image=img)
    imgtk2 = ImageTk.PhotoImage(image=img2)
    lmain.imgtk = imgtk
    lmain.imgtk = imgtk2
    lmain.configure(image=imgtk)
    lmain.configure(image=imgtk2)
    lmain2.imgtk = imgtk
    lmain2.imgtk = imgtk2
    lmain2.configure(image=imgtk)
    lmain2.configure(image=imgtk2)
root.mainloop()
cap.release()
out.release()
cv2.destroyAllWindows()
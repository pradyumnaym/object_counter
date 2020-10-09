import cv2
video=cv2.VideoCapture("campus_better.mp4")
ok,frame=video.read()
out = cv2.VideoWriter('campus_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30,
                      (frame.shape[1], frame.shape[0]))
out.write(frame)
while True:
    ok,frame=video.read()
    if not ok:
        break
    out.write(frame)

out.release()
video.release()
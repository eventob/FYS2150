"""
Dette programmet spiller av en video frame by frame, ved å legge til hver
enkelt frame i en liste, for deretter å loope gjennom den manuelt.
l = framover
j = bakover
q = quit
"""
import cv2

cap = cv2.VideoCapture("stålkuleiolje.mp4")  # støtter .mp4 og .avi
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frame_count / fps

print("--------Video info--------")
print("fps = {:.3} 1/s\ntot frames = {} \nDuration = {:.3} s".format(fps, frame_count, duration))

check = True

frame_list = []
while check:
    """
    read() returnerer True helt til siste frame, og den returnerer data for hvert enkelt frame
    """
    check, frame = cap.read()
    frame_list.append(frame)

"""
Siste verdien i frame list er None, som vi må derfor fjerne pop()
"""
frame_list.pop()
frame_num = 0

while frame_num < frame_count:
    """
    Loopen gjør det mulig å manuelt manøvrere seg gjennom frames
    """
    cv2.imshow("Frame", frame_list[frame_num])
    print(frame_num)

    key = cv2.waitKey(0)

    while key not in [ord('q'), ord('l'), ord("j")]:
        key = cv2.waitKey(0)
    if key == ord("l"):
        frame_num += 1
    if key == ord("j") and frame_num != 0:
        frame_num -= 1
    # Quit when 'q' is pressed
    if key == ord('q'):
        break

"""
Når videoen er ferdig avspilt, lukkes vinduet automatisk
"""
cap.release()
cv2.destroyAllWindows()

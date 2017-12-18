# import cv2
# import numpy as np
#
#
# path='parking cctv.mp4'
#
# vidcap = cv2.VideoCapture(0)
# success,image = vidcap.read()
# count = 0
# while success:
#     success,image = vidcap.read()
#     cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file
#     count += 1
import webbrowser

f = open('helloworld.html','w')

message = """<html>
<head></head>
<body><p>Hello World!</p></body>
</html>"""

f.write(message)
f.close()

webbrowser.open_new_tab('helloworld.html')
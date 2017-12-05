#!/usr/bin/env python
#
from flask import Flask, render_template, Response
from controller_facenet import VideoCamera

app = Flask(__name__)

@app.route('/facenet.html')
def facenet():
    return render_template('facenet.html')

def gen_facenet(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed_facenet')
def video_feed_facenet():
    return Response(gen_facenet(VideoCamera()), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True)
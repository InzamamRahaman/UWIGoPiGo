from picamera import PiCamera
from time import sleep


camera = PiCamera()
camera.capture('/tmp/tmp.jpeg')
camera.close()

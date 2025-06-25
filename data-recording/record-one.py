from metavision_core.event_io import EventsIterator
from metavision_core.event_io.raw_reader import initiate_device
import metavision_hal
import socket
import sys
import os
import time
from opencvbycallback import CameraController
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# event camera parameters
queryInterval = 0.01
startx, starty, widthx, hieghty = 410, 0, 500, 600
bias_diff_on = 20
bias_diff_off = 20

# rgb camera parameters
rgbCameraExposureTime = 25000
rgbCameraMaxWidth = 1280
rgbCameraMaxHeight = 1024
rgbCameraWidth = 600
rgbCameraHeight = 600
rgbCameraXOffset = 264
rgbCameraYOffset = 0

# other parameters
waitTimeAfterEndRotateSignal = 2
inf = 100000


def recv(sock, buffer_size=1024):
    data = sock.recv(buffer_size)
    return data.decode().strip()


def sendStartSignal():
    ip = "192.168.0.11"
    s = socket.socket()
    s.connect((ip, 29999))  # 29999 is the Dashboard port for Universal Robots

    data = recv(s)
    assert data == 'Connected: Universal Robots Dashboard Server'

    s.sendall(b'play\n')  # let the robot start the program
    data = recv(s)
    assert data == 'Starting program'

    s.close()
    logging.info('end sendStartSignal')


class EventCamera:
    def __init__(self, filePath):
        logging.info('start EventCamera __init__')
        device = initiate_device("")
        # setbias
        device.get_i_ll_biases().set("bias_diff_on", bias_diff_on)
        device.get_i_ll_biases().set("bias_diff_off", bias_diff_off)
        # setROI
        windowroi = metavision_hal.I_ROI.Window(startx, starty, widthx, hieghty)
        device.get_i_roi().enable(1)
        device.get_i_roi().set_window(windowroi)

        if device.get_i_events_stream():
            device.get_i_events_stream().log_raw_data(filePath)
        self.device = device
        logging.info('end EventCamera __init__')

    def startCamera(self):
        logging.info('start EventCamera startCamera')
        device = self.device
        delta_t = queryInterval * 1e6
        mv_iterator = EventsIterator.from_device(device=device, delta_t=delta_t)
        self.mv_iterator = mv_iterator
        logging.info('end EventCamera startCamera')
        return mv_iterator

    def endCamera(self):
        logging.info('start EventCamera endCamera')
        # events will be recorded automatically but we can also process/display them
        self.device.get_i_events_stream().stop_log_raw_data()
        logging.info('end EventCamera endCamera')


class RGBCamera:
    def __init__(self, folderPath):
        logging.info('start RGBCamera __init__')
        camera_index = '1'

        controller = CameraController(dir=folderPath)
        self.controller = controller

        try:
            controller.enumerate_devices()
            result = controller.start_camera(camera_index, rgbCameraExposureTime)
            if not result:
                raise Exception("Failed to start camera")

        except Exception as e:
            controller.stop_camera()
            raise e

        logging.info('end RGBCamera __init__')

    def startCamera(self):
        logging.info('start RGBCamera startCamera')
        self.controller.startGrabbing()
        logging.info('end RGBCamera startCamera')

    def endCamera(self):
        logging.info('start RGBCamera endCamera')
        self.controller.stop_camera()
        logging.info('end RGBCamera endCamera')

    def saveAllImage(self):
        maxSize = rgbCameraMaxWidth, rgbCameraMaxHeight
        roi = rgbCameraXOffset, rgbCameraYOffset, rgbCameraWidth, rgbCameraHeight
        self.controller.saveAllImage(maxSize, roi)


def receiveData(connect, blocking):
    connect.setblocking(blocking)
    try:
        data = connect.recv(1024)
    except BlockingIOError:
        return ''

    dataDecode = data.decode().strip()
    logging.info(f'dataDecode: {dataDecode}')
    return dataDecode


def main():
    if len(sys.argv) != 3:
        print("Usage: python record-one.py <eventFilePath> <rgbFolderPath>")
        sys.exit(1)

    eventFilePath = sys.argv[1]
    rgbFolderPath = sys.argv[2]
    logging.info(f'eventFilePath: {eventFilePath}   rgbFolderPath: {rgbFolderPath}')

    # if exist not empty folder, raise error
    if os.path.exists(rgbFolderPath) and os.listdir(rgbFolderPath):
        logging.error(f"Folder {rgbFolderPath} already exists")
        raise FileExistsError(f"Folder {rgbFolderPath} already exists")
    os.makedirs(rgbFolderPath, exist_ok=True)

    sendStartSignal()

    HOST = ''  # listen on all addresses
    PORT = 5000

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((HOST, PORT))
    s.listen(1)

    logging.info('waiting for connection')
    conn, addr = s.accept()
    logging.info(f'connection established with {addr}')

    data = receiveData(conn, True)
    assert data == 'start program'

    # start event camera and RGB camera
    eventCamera = EventCamera(eventFilePath)
    rgbCamera = RGBCamera(rgbFolderPath)

    # wait for start rotate signal
    startReceiveTime = time.time()
    data = receiveData(conn, True)
    endReceiveTime = time.time()

    # if sleep time is too short, it may not receive the signal in time
    if endReceiveTime - startReceiveTime < 1:
        logging.error('not receive start rotate signal in time')
        exit(55)
    assert data == 'start rotate'

    # start RGB camera first, because it takes longer to start
    rgbCamera.startCamera()
    mv_iterator = eventCamera.startCamera()

    # start event camera and RGB camera
    stopIndex = inf

    for i, evs in enumerate(mv_iterator):
        if i >= stopIndex + waitTimeAfterEndRotateSignal / queryInterval:
            eventCamera.endCamera()
            rgbCamera.endCamera()
            break
        if i % 100 == 0:
            logging.info(f'recording {i*queryInterval} s')

        data = receiveData(conn, False)
        assert data == '' or data == 'end rotate'
        if data == 'end rotate':
            stopIndex = i

    # save rgb images
    time.sleep(2)
    rgbCamera.saveAllImage()

    # receive end program signal
    data = receiveData(conn, True)
    assert data == 'end program'

    conn.close()
    logging.info('connection closed')


main()

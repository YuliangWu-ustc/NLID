import sys
from ctypes import *
import datetime
import numpy
import cv2
import time
import os
import logging

path = os.getcwd()
os.chdir('/opt/HuarayTech/MVviewer/share/Python/MVSDK/')
sys.path.append("/opt/HuarayTech/MVviewer/share/Python/MVSDK/")

from IMVApi import *

os.chdir(path)


class CameraController:
    def __init__(self, dir=None):
        self.cam = None
        self.deviceList = None
        self.CALL_BACK_FUN = None
        self.initialize_callback()
        self.savedir = dir
        self.imgnum = 0
        self.frametime = None
        self.allColorByteArray = []
        self.allWidthHeight = []
        self.setExposureTime = False
        self.prevImageTime = None

    def initialize_callback(self):
        pFrame = POINTER(IMV_Frame)
        FrameInfoCallBack = eval('CFUNCTYPE')(None, pFrame, c_void_p)
        self.CALL_BACK_FUN = FrameInfoCallBack(self.image_callback)

    def setframetime(self, frametime):
        self.frametime = frametime

    def image_callback(self, pFrame, pUser):
        startTime = datetime.datetime.now()
        if self.prevImageTime is None:
            logging.debug('first image_callback')
            self.firstImageTime = startTime
        else:
            duration = round((startTime - self.prevImageTime).total_seconds() * 1000)
            logging.debug(f'start image_callback   duration: {duration} ms')
        self.prevImageTime = startTime

        if pFrame is None:
            raise ValueError("pFrame is NULL")

        stPixelConvertParam = IMV_PixelConvertParam()
        frame = cast(pFrame, POINTER(IMV_Frame)).contents
        blockId = frame.frameInfo.blockId

        # calculate destination buffer size
        nDstBufSize = (frame.frameInfo.width * frame.frameInfo.height *
                       (1 if IMV_EPixelType.gvspPixelMono8 == frame.frameInfo.pixelFormat else 3))

        pDstBuf = (c_ubyte * nDstBufSize)()
        memset(byref(stPixelConvertParam), 0, sizeof(stPixelConvertParam))

        # set pixel convert parameters
        self._setup_pixel_convert_param(stPixelConvertParam, frame, pDstBuf, nDstBufSize)

        # release the frame
        nRet = self.cam.IMV_ReleaseFrame(frame)
        if IMV_OK != nRet:
            raise Exception(f"Release frame failed! ErrorCode[{nRet}]")

        # process the image
        self.appendColorImageToByteArray(stPixelConvertParam)

        endTime = datetime.datetime.now()
        functionTime = round((endTime - startTime).total_seconds() * 1000)
        logging.debug(f'end image_callback   blockId: {blockId}   functionTime: {functionTime} ms')

    def _setup_pixel_convert_param(self, param, frame, pDstBuf, nDstBufSize):
        param.nWidth = frame.frameInfo.width
        param.nHeight = frame.frameInfo.height
        param.ePixelFormat = frame.frameInfo.pixelFormat
        param.pSrcData = frame.pData
        param.nSrcDataLen = frame.frameInfo.size
        param.nPaddingX = frame.frameInfo.paddingX
        param.nPaddingY = frame.frameInfo.paddingY
        param.eBayerDemosaic = IMV_EBayerDemosaic.demosaicNearestNeighbor
        param.eDstPixelFormat = frame.frameInfo.pixelFormat
        param.pDstBuf = pDstBuf
        param.nDstBufSize = nDstBufSize

    def _convert_to_cv_image(self, param):
        if param.ePixelFormat == IMV_EPixelType.gvspPixelMono8:
            return self._convert_mono8(param)
        else:
            return self._convert_color(param)

    def _convert_mono8(self, param):
        # process mono8 image
        userBuff = c_buffer(b'\0', param.nDstBufSize)
        memmove(userBuff, param.pSrcData, param.nDstBufSize)
        grayByteArray = bytearray(userBuff)
        return numpy.array(grayByteArray).reshape(param.nHeight, param.nWidth)

    def appendColorImageToByteArray(self, param):
        param.eDstPixelFormat = IMV_EPixelType.gvspPixelBGR8
        nRet = self.cam.IMV_PixelConvert(param)
        if IMV_OK != nRet:
            raise Exception(f"Image convert failed! ErrorCode[{nRet}]")

        rgbBuff = c_buffer(b'\0', param.nDstBufSize)
        memmove(rgbBuff, param.pDstBuf, param.nDstBufSize)
        colorByteArray = bytearray(rgbBuff)
        self.allWidthHeight.append((param.nWidth, param.nHeight))
        self.allColorByteArray.append(colorByteArray)

    def saveAllImage(self, maxSize, roi):
        imageLength = len(self.allColorByteArray)
        duration = (self.prevImageTime - self.firstImageTime).total_seconds()
        fps = (imageLength - 1) / duration
        logging.info(f'start saveAllImage   imageLength: {imageLength}   duration: {duration:.3f} s   fps: {fps:.1f}')

        if self.setExposureTime:
            raise Exception(
                'After setting the exposure time, you need to capture some images for correct shooting. Also, you need to manually enable the frame rate and set it to 30 fps.'
            )

        # check allWidthHeight have same width and height
        if len(self.allWidthHeight) == 0:
            raise ValueError("No images to save")
        if len(set(self.allWidthHeight)) != 1:
            raise ValueError("All images must have the same width and height")

        nWidth, nHeight = self.allWidthHeight[0]
        logging.info(f'nWidth: {nWidth}   nHeight: {nHeight}')
        if (nWidth, nHeight) != maxSize:
            raise ValueError(f"Image size mismatch: expected {maxSize}, got {(nWidth, nHeight)}")

        for colorByteArray in self.allColorByteArray:
            cvImage = numpy.array(colorByteArray).reshape(nHeight, nWidth, 3)
            x, y, w, h = roi
            cvImage = cvImage[y:y + h, x:x + w]

            cv2.imwrite(os.path.join(self.savedir, f"{self.imgnum:08d}.jpg"), cvImage)
            self.imgnum += 1
        logging.info('end saveAllImage')

    def setCameraExposureTime(self, exposureTime):
        cam = self.cam
        initExposureTime = c_double(0.0)
        exposureTimeValue = c_double(exposureTime)
        finalExposureTime = c_double(0.0)

        # get current exposure time
        ret = cam.IMV_GetDoubleFeatureValue("ExposureTime", initExposureTime)
        if ret != IMV_OK:
            raise Exception(f"Get current exposure time failed! ErrorCode {ret}")
        logging.info(f"Current exposure time: {initExposureTime.value}")

        if abs(initExposureTime.value - exposureTime) > 0.1:
            # set exposure time
            ret = cam.IMV_SetDoubleFeatureValue("ExposureTime", exposureTimeValue.value)
            if ret != IMV_OK:
                raise Exception(f"Set exposure time failed! ErrorCode {ret}")

            # get new exposure time
            ret = cam.IMV_GetDoubleFeatureValue("ExposureTime", finalExposureTime)
            if ret != IMV_OK:
                raise Exception(f"Get new exposure time failed! ErrorCode {ret}")
            logging.info(f"New exposure time: {finalExposureTime.value}")

            if abs(finalExposureTime.value - exposureTime) > 0.1:
                raise Exception(f"Failed to set exposure time to {exposureTime}")
            self.setExposureTime = True

    def enumerate_devices(self):
        self.deviceList = IMV_DeviceList()
        nRet = MvCamera.IMV_EnumDevices(self.deviceList, IMV_EInterfaceType.interfaceTypeAll)
        if IMV_OK != nRet:
            raise Exception(f"Enumeration devices failed! ErrorCode {nRet}")
        if self.deviceList.nDevNum == 0:
            raise Exception("No devices found!")
        return self.deviceList

    def display_device_info(self):
        if not self.deviceList:
            self.enumerate_devices()

        print("Idx  Type   Vendor              Model           S/N                 DeviceUserID    IP Address")
        print("-" * 96)
        for i in range(self.deviceList.nDevNum):
            dev_info = self.deviceList.pDevInfo[i]
            dev_type = "Gige" if dev_info.nCameraType == typeGigeCamera else "U3V"
            print(
                f"[{i+1}]  {dev_type:<6} {dev_info.vendorName.decode('ascii'):<18} "
                f"{dev_info.modelName.decode('ascii'):<15} {dev_info.serialNumber.decode('ascii'):<20} "
                f"{dev_info.cameraName.decode('ascii'):<15} {dev_info.DeviceSpecificInfo.gigeDeviceInfo.ipAddress.decode('ascii')}"
            )

    def start_camera(self, camera_index, exposureTime):
        try:
            self.cam = MvCamera()

            # create camera handle
            nRet = self.cam.IMV_CreateHandle(IMV_ECreateHandleMode.modeByIndex, byref(c_void_p(int(camera_index) - 1)))
            if IMV_OK != nRet:
                raise Exception(f"Create handle failed! ErrorCode {nRet}")

            # open camera
            nRet = self.cam.IMV_Open()
            if IMV_OK != nRet:
                raise Exception(f"Open camera failed! ErrorCode {nRet}")

            # set exposure time and trigger mode
            self.setCameraExposureTime(exposureTime)
            self._setup_trigger_mode()

            # attach grabbing callback
            nRet = self.cam.IMV_AttachGrabbing(self.CALL_BACK_FUN, None)
            if IMV_OK != nRet:
                raise Exception(f"Attach grabbing failed! ErrorCode {nRet}")

            return True

        except Exception as e:
            print(f"Error starting camera: {str(e)}")
            self.stop_camera()
            raise e

    def startGrabbing(self):
        nRet = self.cam.IMV_StartGrabbing()
        if IMV_OK != nRet:
            raise Exception(f"Start grabbing failed! ErrorCode {nRet}")

    def _setup_trigger_mode(self):
        trigger_settings = [("TriggerSource", "Software"), ("TriggerSelector", "FrameStart"), ("TriggerMode", "Off")]

        for setting, value in trigger_settings:
            nRet = self.cam.IMV_SetEnumFeatureSymbol(setting, value)
            if IMV_OK != nRet:
                raise Exception(f"Set {setting} failed! ErrorCode {nRet}")

    def stop_camera(self):
        if self.cam:
            self.cam.IMV_StopGrabbing()
            self.cam.IMV_Close()
            if self.cam.handle:
                self.cam.IMV_DestroyHandle()
            self.cam = None

    def run_capture(self, duration=10):
        time.sleep(duration)


def main():
    imageFolder = 'images'
    os.makedirs(imageFolder, exist_ok=True)
    controller = CameraController(imageFolder)
    try:
        controller.enumerate_devices()
        controller.display_device_info()

        camera_index = input("Please input the camera index: ")
        if controller.start_camera(camera_index):
            controller.run_capture(10)
    finally:
        controller.stop_camera()
        print("---Demo end---")


if __name__ == "__main__":
    main()

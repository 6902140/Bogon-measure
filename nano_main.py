import numpy as np
from pyorbbecsdk import Pipeline,Config,OBSensorType,FrameSet,OBFormat,OBPropertyID,OBAlignMode
import onnxruntime as ort
from PyQt5 import QtWidgets,QtGui
from socket import socket, AF_INET, SOCK_STREAM
from enum import Enum
import os
import threading
import sys
import cv2
import math
import pickle
import time

SERVER_IP = '127.0.0.1'
SERVER_PORT = 6666
SOCKET_TIMEOUT=5
MODEL_PATH='/home/bogon/Desktop/model.onnx'
SAVE_DIR='/home/bogon/Desktop/result_m'
PICKLES_DIR='/home/bogon/Desktop/pickles'
STAGE1_BRIGHTNESS=100
STAGE2_BRIGHTNESS=100
CONF=0.25
IOU=0.7

class YOLOSeg:
    """YOLOv8 segmentation model."""

    def __init__(self, onnx_model):
        """
        Initialization.

        Args:
            onnx_model (str): Path to the ONNX model.
        """

        # Build Ort session
        self.session = ort.InferenceSession(
            onnx_model,
            providers=["CPUExecutionProvider"],
        )

        # Numpy dtype: support both FP32 and FP16 onnx model
        self.ndtype = np.half if self.session.get_inputs()[0].type == "tensor(float16)" else np.single

        # Get model width and height(YOLOv8-seg only has one input)
        self.model_height, self.model_width = [x.shape for x in self.session.get_inputs()][0][-2:]

        self.classes={0:'inside',1:'gasket',2:'screw',3:'desktop'}

    def __call__(self, im0, conf_threshold=0.4, iou_threshold=0.45, nm=32):
        """
        The whole pipeline: pre-process -> inference -> post-process.

        Args:
            im0 (Numpy.ndarray): original input image.
            conf_threshold (float): confidence threshold for filtering predictions.
            iou_threshold (float): iou threshold for NMS.
            nm (int): the number of masks.

        Returns:
            boxes (List): list of bounding boxes.
            segments (List): list of segments.
            masks (np.ndarray): [N, H, W], output masks.
        """

        # Pre-process
        im, ratio, (pad_w, pad_h) = self.preprocess(im0)

        # Ort inference
        preds = self.session.run(None, {self.session.get_inputs()[0].name: im})

        # Post-process
        boxes, segments, masks = self.postprocess(
            preds,
            im0=im0,
            ratio=ratio,
            pad_w=pad_w,
            pad_h=pad_h,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            nm=nm,
        )
        return boxes, segments, masks

    def preprocess(self, img):
        """
        Pre-processes the input image.

        Args:
            img (Numpy.ndarray): image about to be processed.

        Returns:
            img_process (Numpy.ndarray): image preprocessed for inference.
            ratio (tuple): width, height ratios in letterbox.
            pad_w (float): width padding in letterbox.
            pad_h (float): height padding in letterbox.
        """

        # Resize and pad input image using letterbox() (Borrowed from Ultralytics)
        shape = img.shape[:2]  # original image shape
        new_shape = (self.model_height, self.model_width)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        ratio = r, r
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        pad_w, pad_h = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2  # wh padding
        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
        left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

        # Transforms: HWC to CHW -> BGR to RGB -> div(255) -> contiguous -> add axis(optional)
        img = np.ascontiguousarray(np.einsum("HWC->CHW", img)[::-1], dtype=self.ndtype) / 255.0
        img_process = img[None] if len(img.shape) == 3 else img
        return img_process, ratio, (pad_w, pad_h)

    def postprocess(self, preds, im0, ratio, pad_w, pad_h, conf_threshold, iou_threshold, nm=32):
        """
        Post-process the prediction.

        Args:
            preds (Numpy.ndarray): predictions come from ort.session.run().
            im0 (Numpy.ndarray): [h, w, c] original input image.
            ratio (tuple): width, height ratios in letterbox.
            pad_w (float): width padding in letterbox.
            pad_h (float): height padding in letterbox.
            conf_threshold (float): conf threshold.
            iou_threshold (float): iou threshold.
            nm (int): the number of masks.

        Returns:
            boxes (List): list of bounding boxes.
            segments (List): list of segments.
            masks (np.ndarray): [N, H, W], output masks.
        """
        x, protos = preds[0], preds[1]  # Two outputs: predictions and protos

        # Transpose the first output: (Batch_size, xywh_conf_cls_nm, Num_anchors) -> (Batch_size, Num_anchors, xywh_conf_cls_nm)
        x = np.einsum("bcn->bnc", x)

        # Predictions filtering by conf-threshold
        x = x[np.amax(x[..., 4:-nm], axis=-1) > conf_threshold]

        # Create a new matrix which merge these(box, score, cls, nm) into one
        # For more details about `numpy.c_()`: https://numpy.org/doc/1.26/reference/generated/numpy.c_.html
        x = np.c_[x[..., :4], np.amax(x[..., 4:-nm], axis=-1), np.argmax(x[..., 4:-nm], axis=-1), x[..., -nm:]]

        # NMS filtering
        x = x[cv2.dnn.NMSBoxes(x[:, :4], x[:, 4], conf_threshold, iou_threshold)]

        # Decode and return
        if len(x) > 0:
            # Bounding boxes format change: cxcywh -> xyxy
            x[..., [0, 1]] -= x[..., [2, 3]] / 2
            x[..., [2, 3]] += x[..., [0, 1]]

            # Rescales bounding boxes from model shape(model_height, model_width) to the shape of original image
            x[..., :4] -= [pad_w, pad_h, pad_w, pad_h]
            x[..., :4] /= min(ratio)

            # Bounding boxes boundary clamp
            x[..., [0, 2]] = x[:, [0, 2]].clip(0, im0.shape[1])
            x[..., [1, 3]] = x[:, [1, 3]].clip(0, im0.shape[0])

            # Process masks
            masks = self.process_mask(protos[0], x[:, 6:], x[:, :4], im0.shape)

            # Masks -> Segments(contours)
            segments = self.masks2segments(masks)
            return x[..., :6], segments, masks  # boxes, segments, masks
        else:
            return [], [], []

    @staticmethod
    def masks2segments(masks):
        """
        It takes a list of masks(n,h,w) and returns a list of segments(n,xy) (Borrowed from
        https://github.com/ultralytics/ultralytics/blob/465df3024f44fa97d4fad9986530d5a13cdabdca/ultralytics/utils/ops.py#L750)

        Args:
            masks (numpy.ndarray): the output of the model, which is a tensor of shape (batch_size, 160, 160).

        Returns:
            segments (List): list of segment masks.
        """
        segments = []
        for x in masks.astype("uint8"):
            c = cv2.findContours(x, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]  # CHAIN_APPROX_SIMPLE
            if c:
                c = np.array(c[np.array([len(x) for x in c]).argmax()]).reshape(-1, 2)
            else:
                c = np.zeros((0, 2))  # no segments found
            segments.append(c.astype("float32"))
        return segments

    @staticmethod
    def crop_mask(masks, boxes):
        """
        It takes a mask and a bounding box, and returns a mask that is cropped to the bounding box. (Borrowed from
        https://github.com/ultralytics/ultralytics/blob/465df3024f44fa97d4fad9986530d5a13cdabdca/ultralytics/utils/ops.py#L599)

        Args:
            masks (Numpy.ndarray): [n, h, w] tensor of masks.
            boxes (Numpy.ndarray): [n, 4] tensor of bbox coordinates in relative point form.

        Returns:
            (Numpy.ndarray): The masks are being cropped to the bounding box.
        """
        n, h, w = masks.shape
        x1, y1, x2, y2 = np.split(boxes[:, :, None], 4, 1)
        r = np.arange(w, dtype=x1.dtype)[None, None, :]
        c = np.arange(h, dtype=x1.dtype)[None, :, None]
        return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))

    def process_mask(self, protos, masks_in, bboxes, im0_shape):
        """
        Takes the output of the mask head, and applies the mask to the bounding boxes. This produces masks of higher quality
        but is slower. (Borrowed from https://github.com/ultralytics/ultralytics/blob/465df3024f44fa97d4fad9986530d5a13cdabdca/ultralytics/utils/ops.py#L618)

        Args:
            protos (numpy.ndarray): [mask_dim, mask_h, mask_w].
            masks_in (numpy.ndarray): [n, mask_dim], n is number of masks after nms.
            bboxes (numpy.ndarray): bboxes re-scaled to original image shape.
            im0_shape (tuple): the size of the input image (h,w,c).

        Returns:
            (numpy.ndarray): The upsampled masks.
        """
        c, mh, mw = protos.shape
        masks = np.matmul(masks_in, protos.reshape((c, -1))).reshape((-1, mh, mw)).transpose(1, 2, 0)  # HWN
        masks = np.ascontiguousarray(masks)
        masks = self.scale_mask(masks, im0_shape)  # re-scale mask from P3 shape to original input image shape
        masks = np.einsum("HWN -> NHW", masks)  # HWN -> NHW
        masks = self.crop_mask(masks, bboxes)
        return np.greater(masks, 0.5)

    @staticmethod
    def scale_mask(masks, im0_shape, ratio_pad=None):
        """
        Takes a mask, and resizes it to the original image size. (Borrowed from
        https://github.com/ultralytics/ultralytics/blob/465df3024f44fa97d4fad9986530d5a13cdabdca/ultralytics/utils/ops.py#L305)

        Args:
            masks (np.ndarray): resized and padded masks/images, [h, w, num]/[h, w, 3].
            im0_shape (tuple): the original image shape.
            ratio_pad (tuple): the ratio of the padding to the original image.

        Returns:
            masks (np.ndarray): The masks that are being returned.
        """
        im1_shape = masks.shape[:2]
        if ratio_pad is None:  # calculate from im0_shape
            gain = min(im1_shape[0] / im0_shape[0], im1_shape[1] / im0_shape[1])  # gain  = old / new
            pad = (im1_shape[1] - im0_shape[1] * gain) / 2, (im1_shape[0] - im0_shape[0] * gain) / 2  # wh padding
        else:
            pad = ratio_pad[1]

        # Calculate tlbr of mask
        top, left = int(round(pad[1] - 0.1)), int(round(pad[0] - 0.1))  # y, x
        bottom, right = int(round(im1_shape[0] - pad[1] + 0.1)), int(round(im1_shape[1] - pad[0] + 0.1))
        if len(masks.shape) < 2:
            raise ValueError(f'"len of masks shape" should be 2 or 3, but got {len(masks.shape)}')
        masks = masks[top:bottom, left:right]
        masks = cv2.resize(
            masks, (im0_shape[1], im0_shape[0]), interpolation=cv2.INTER_LINEAR
        )  # INTER_CUBIC would be better
        if len(masks.shape) == 2:
            masks = masks[:, :, None]
        return masks

    def draw_and_visualize(self, im, bboxes, segments):
        """
        Draw and visualize results.

        Args:
            im (np.ndarray): original image, shape [h, w, c].
            bboxes (numpy.ndarray): [n, 4], n is number of bboxes.
            segments (List): list of segment masks.
            vis (bool): imshow using OpenCV.
            save (bool): save image annotated.

        Returns:
            None
        """

        # Draw rectangles and polygons
        im_canvas = im.copy()
        for (*box, conf, cls_), segment in zip(bboxes, segments):
            # draw contour and fill mask
            cv2.polylines(im, np.int32([segment]), True, (255,0,0), 5)  # white borderline

            # draw bbox rectangle
            cv2.rectangle(
                im,
                (int(box[0]), int(box[1])),
                (int(box[2]), int(box[3])),
                (255,0,0),
                5,
                cv2.LINE_AA,
            )
            cv2.putText(
                im,
                f"{self.classes[cls_]}: {conf:.3f}",
                (int(box[0]), int(box[1] - 9)),
                cv2.FONT_HERSHEY_SIMPLEX,
                2.5,
                (255,0,0),
               
                5,
                cv2.LINE_AA,
            )

        # Mix image
        im = cv2.addWeighted(im_canvas, 0.3, im, 0.7, 0)

        return im

class DataType(Enum):
    TEAM_ID = 0
    RESULT = 2

def pack_data(type: DataType, text: str) -> bytearray:
    packed = bytearray()
    packed.extend(type.value.to_bytes(4, 'big'))
    data = text.encode("utf-8")
    length = len(data).to_bytes(4, 'big')
    packed.extend(length)
    packed.extend(data)
    return packed

def distance_to_line(k, b, x0, y0):
    numerator = abs(k * x0 - y0 + b)
    denominator = math.sqrt(k**2 + 1)
    distance = numerator / denominator
    return distance

def sort_points_clockwise(points):
    point1 = min(points, key=lambda x: x[1] + 0.8*x[0]) # 找到最左下角的点
    sorted_points = [point1]
    point2=max(points, key=lambda x: x[1] - 0.8*x[0]) # 找到最左下角的点
    sorted_points.append(point2)
    point3=max(points, key=lambda x: x[1] + 0.8*x[0]) # 找到最左下角的点
    sorted_points.append(point3)
    point4=min(points, key=lambda x: x[1] - 0.8*x[0]) # 找到最左下角的点
    sorted_points.append(point4)
    return sorted_points

def outside_rectangle_fit(data):
    rect = cv2.minAreaRect(data)
    box = cv2.boxPoints(rect)
    box = np.round(box)
    box = np.int64(box)
    len1=np.linalg.norm(box[0] - box[1])
    len2=np.linalg.norm(box[0] - box[2])
    len3=np.linalg.norm(box[0] - box[3])
    decision=-1
    if len1>len2:
        if len2>len3:
            decision=2
        else:
            if len1>len3:
                decision=3
            else:
                decision=1
    else:# len1<len2
        if len3>len2:
            decision=2
        else:
            if len1>len3:
                decision=1
            else:
                decision=3
    if box[decision][0]-box[0][0]!=0:   
        k=(box[decision][1]-box[0][1])/(box[decision][0]-box[0][0])
    else:
        k=10000
    if k>1000:
        k=1000
    if k<=-1000:
        k=-1000
    if k<0.001 and k>=0:
        k=0.001
    if k>-0.001 and k<=0:
        k=-0.001
    b=rect[0][1]-k*rect[0][0]
    return rect,k,b

class UI(QtWidgets.QWidget):
    def __init__(self):
        super(UI,self).__init__()
        print("正在初始化界面")
        self.setFixedSize(770,540)
        pixmap = QtGui.QPixmap(320,240)
        pixmap.fill(QtGui.QColor(0))
        pixmap2 = QtGui.QPixmap(320,240)
        pixmap2.fill(QtGui.QColor(0))
        self.label = QtWidgets.QLabel()
        self.label.setPixmap(pixmap)
        self.label2 = QtWidgets.QLabel()
        self.label2.setPixmap(pixmap2)
        self.table_result = QtWidgets.QTableWidget(0,4)
        self.table_result.setFixedSize(402,240)
        self.table_result.setHorizontalHeaderLabels(['Goal_ID', 'Goal_A', 'Goal_B','Goal_C'])
        self.table_result.setRowCount(0)
        self.button_detect = QtWidgets.QPushButton()
        self.button_detect.setFixedSize(410,40)
        self.button_detect.setText("开始识别")
        self.button_detect.clicked.connect(self.on_button_detect_clicked)
        self.radio1 = QtWidgets.QRadioButton('静态测量') 
        self.radio1.setChecked(True)
        self.radio1.toggled.connect(self.on_radio_toggled1)
        self.radio2 = QtWidgets.QRadioButton('动态测量')
        self.radio2.setChecked(False)
        self.radio2.toggled.connect(self.on_radio_toggled2)
        self.vbox = QtWidgets.QVBoxLayout()
        self.vbox.addWidget(self.radio1)
        self.vbox.addWidget(self.radio2)
        self.label_status = QtWidgets.QLabel()
        self.label_status.setText("就绪")
        self.gLayout = QtWidgets.QGridLayout()
        self.gLayout.addWidget(self.label,0,0,4,1)
        self.gLayout.addWidget(self.label2,4,0,4,1)
        self.gLayout.addWidget(self.table_result,0,1,4,1)
        self.gLayout.addLayout(self.vbox,4,1,2,1)
        self.gLayout.addWidget(self.button_detect,6,1,1,1)
        self.gLayout.addWidget(self.label_status,8,0,1,2)
        self.setLayout(self.gLayout)

        print("正在加载模型")
        self.model=YOLOSeg(MODEL_PATH)
        with open(os.path.join(PICKLES_DIR, 'square_gasket.pkl'), 'rb') as f:
            self.pkl_square_gasket = pickle.load(f)
        with open(os.path.join(PICKLES_DIR, 'square_screw_diameter.pkl'), 'rb') as f:
            self.pkl_square_screw_diameter = pickle.load(f)
        with open(os.path.join(PICKLES_DIR, 'square_screw_length.pkl'), 'rb') as f:
            self.pkl_square_screw_length = pickle.load(f)
        with open(os.path.join(PICKLES_DIR, 'round_gasket.pkl'), 'rb') as f:
            self.pkl_round_gasket = pickle.load(f)
        with open(os.path.join(PICKLES_DIR, 'round_screw_diameter.pkl'), 'rb') as f:
            self.pkl_round_screw_diameter = pickle.load(f)
        with open(os.path.join(PICKLES_DIR, 'round_screw_length.pkl'), 'rb') as f:
            self.pkl_round_screw_length = pickle.load(f)

        print("正在连接相机")
        self.mode=1
        self.camera_thread=threading.Thread(target=self.camera_thread_run)
        self.predict_running=False
        self.currentFrame=None
        self.socket = socket(AF_INET, SOCK_STREAM)
        self.socket.settimeout(SOCKET_TIMEOUT)
        try:
            self.socket.connect((SERVER_IP, SERVER_PORT))
        except Exception as e:
            print(e)
        self.pipeline=Pipeline()
        self.device=self.pipeline.get_device()
        self.config=Config()

        print("正在初始化相机")
        self.device.set_bool_property(OBPropertyID.OB_PROP_COLOR_AUTO_EXPOSURE_BOOL, False)
        if self.mode==1:
            self.device.set_int_property(OBPropertyID.OB_PROP_COLOR_EXPOSURE_INT, STAGE1_BRIGHTNESS)
        else:
            self.device.set_int_property(OBPropertyID.OB_PROP_COLOR_EXPOSURE_INT, STAGE2_BRIGHTNESS)
        try:
            profile_list = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            color_profile = profile_list.get_video_stream_profile(2048,0,OBFormat.RGB,15)
            self.config.enable_stream(color_profile)
            self.pipeline.start(self.config)
        except Exception as e:
            print(e)
            exit()
        self.camera_running=True
        self.label_status.setText("运行中")
        self.camera_thread.start()

    def on_radio_toggled1(self, checked):  
        if checked:  
            self.radio2.setChecked(False)
            self.mode=1
        self.update_camera()
    
    def on_radio_toggled2(self, checked):  
        if checked:  
            self.radio1.setChecked(False)
            self.mode=2  
        self.update_camera()

    def on_button_detect_clicked(self):
        if not self.predict_running:
            threading.Thread(target=self.predict_thread_run).start()
            self.predict_running=True


    def update_camera(self):
        self.label_status.setText("正在初始化相机")
        self.camera_running=False
        self.pipeline.stop()
        self.device.set_bool_property(OBPropertyID.OB_PROP_COLOR_AUTO_EXPOSURE_BOOL, False)
        if self.mode==1:
            self.device.set_int_property(OBPropertyID.OB_PROP_COLOR_EXPOSURE_INT, STAGE1_BRIGHTNESS)
        else:
            self.device.set_int_property(OBPropertyID.OB_PROP_COLOR_EXPOSURE_INT, STAGE2_BRIGHTNESS)
        try:
            self.pipeline.start(self.config)
        except Exception as e:
            print(e)
            return
        self.camera_running=True
        self.label_status.setText("运行中")

    def camera_thread_run(self):
        while True:
            if not self.camera_running:
                continue
            frames: FrameSet = self.pipeline.wait_for_frames(100)
            if frames is None:
                continue
            color_frame = frames.get_color_frame()
            if color_frame is None:
                continue
            width = color_frame.get_width()
            height = color_frame.get_height()
            data = np.asanyarray(color_frame.get_data())
            color_image = np.resize(data, (height, width, 3))
            if color_image is None:
                print("failed to convert frame to image")
                continue
            self.currentFrame=color_image
            showImage = QtGui.QImage(self.currentFrame.astype(np.uint8),self.currentFrame.shape[1],self.currentFrame.shape[0],QtGui.QImage.Format_RGB888)
            self.label.setPixmap(QtGui.QPixmap.fromImage(showImage).scaled(320,240))

    def predict_thread_run(self):
        self.label_status.setText("正在识别目标")
        try:
            self.socket.send(pack_data(DataType.TEAM_ID, "xjtu"))
        except Exception as e:
            print(e)
        try:
            time0=time.time()
            frame=self.currentFrame
            boxes, segments, _=self.model(frame,conf_threshold=CONF,iou_threshold=IOU)
            nums=len(boxes)
            screw_res=[]
            gasket_res=[]
            inside_res=[]
            combine_res=[]
            center_dis=np.finfo(np.float32).max
            for i in range(nums):
                if int(boxes[i][5])==3:
                    if self.mode==1:
                        approx=None
                        points=np.array(segments[i])
                        x,y,w,h=cv2.boundingRect(points)
                        outside_rect=cv2.minAreaRect(points)
                        dis=np.sqrt((x-640)**2+(y-480)**2)
                        if(dis>center_dis):
                            continue
                        else:
                            center_dis=dis
                        points_int = np.round(points).astype(np.int32)
                        initial_val=0.001
                        while True:
                            epsilon = initial_val * cv2.arcLength(points_int, True)
                            approx = cv2.approxPolyDP(points_int, epsilon, True) #  a N x 1 x 2 NumPy
                            if len(approx) == 4:
                                break
                            elif len(approx)>=4:
                                initial_val=initial_val*1.1
                                continue
                            else:
                                initial_val=initial_val*0.9
                        four_vortex=[]
                        for vortex_index in range(4):
                            four_vortex.append(approx[vortex_index][0])
                        four_vortex=sort_points_clockwise(four_vortex)
                        standard_vortex=[[256,256],[256,768],[768,768],[768,256]]
                        four_vortex = np.array(four_vortex, dtype=np.float32)
                        standard_vortex = np.array(standard_vortex, dtype=np.float32)
                        matrix = cv2.getPerspectiveTransform(four_vortex, standard_vortex)
                    else:
                        points=np.array(segments[i])
                        x,y,w,h=cv2.boundingRect(points)
                        outside_rect=cv2.minAreaRect(points)
                        dis=np.sqrt((x-640)**2+(y-480)**2)
                        if(dis>center_dis):
                            continue
                        else:
                            center_dis=dis
                        outside_box = cv2.boxPoints(outside_rect)
                        outside_box = np.round(outside_box)
                        outside_box = np.int64(outside_box)
                        outside_dot=sort_points_clockwise(outside_box)
                        standard_vortex=[[256,256],[256,768],[768,768],[768,256]]
                        standard_vortex = np.array(standard_vortex, dtype=np.float32)
                        four_vortex = np.array(outside_dot, dtype=np.float32)
                        matrix = cv2.getPerspectiveTransform(four_vortex, standard_vortex)
            for i in range(nums):    
                if int(boxes[i][5])==0 or int(boxes[i][5])==1:
                    points=np.array(segments[i])
                    points_trans=cv2.perspectiveTransform(points.reshape(-1, 1, 2), matrix)
                    ellipse=cv2.fitEllipse(points)
                    ellipse_trans=cv2.fitEllipse(points_trans)
                    if self.mode==1:
                        if ellipse_trans[0][0]<=512 and ellipse_trans[0][0]>=256 and ellipse_trans[0][1]<=512 and ellipse_trans[0][1]>=256:
                            pos=1
                        elif ellipse_trans[0][0]>=512 and ellipse_trans[0][0]<=768 and ellipse_trans[0][1]<=512 and ellipse_trans[0][1]>=256:
                            pos=2
                        elif ellipse_trans[0][0]>=512 and ellipse_trans[0][0]<=768 and ellipse_trans[0][1]>=512 and ellipse_trans[0][1]<=768:
                            pos=3
                        elif ellipse_trans[0][0]<=512 and ellipse_trans[0][0]>=256 and ellipse_trans[0][1]>=512 and ellipse_trans[0][1]<=768:
                            pos=4
                        else:
                            continue
                    else:
                        distance=math.sqrt((ellipse_trans[0][0]-512)**2+(ellipse_trans[0][1]-512)**2)
                        if distance<=256/3:
                            pos=5
                        elif distance>256/3 and distance<=256/3*2:
                            pos=6
                        elif distance>256/3*2 and distance<256:
                            pos=7
                        else:
                            continue
                    X=[h/w,outside_rect[0][1]-ellipse[0][1],max(ellipse_trans[1])]
                    if self.mode==1:
                        Y=self.pkl_square_gasket.predict([X,])[0]
                    else:
                        Y=self.pkl_round_gasket.predict([X],)[0]
                    if int(boxes[i][5])==0:
                        inside_res.append([Y,pos,ellipse_trans[0]])
                    else:
                        gasket_res.append([Y,pos,ellipse_trans[0]])

                if int(boxes[i][5])==2:
                    points=np.array(segments[i])
                    rect,k_fit,b_fit=outside_rectangle_fit(points)
                    points_trans=cv2.perspectiveTransform(points.reshape(-1, 1, 2), matrix)
                    rect_trans,k,b = outside_rectangle_fit(points_trans)
                    offset=(1-h/w)*30
                    if self.mode==1:
                        if rect_trans[0][0]<=512 and rect_trans[0][0]>=256 and rect_trans[0][1]+offset<=512 and rect_trans[0][1]+offset>=256:
                            pos=1
                        elif rect_trans[0][0]>=512 and rect_trans[0][0]<=768 and rect_trans[0][1]+offset<=512 and rect_trans[0][1]+offset>=256:
                            pos=2
                        elif rect_trans[0][0]>=512 and rect_trans[0][0]<=768 and rect_trans[0][1]+offset>=512 and rect_trans[0][1]+offset<=768:
                            pos=3
                        elif rect_trans[0][0]<=512 and rect_trans[0][0]>=256 and rect_trans[0][1]+offset>=512 and rect_trans[0][1]+offset<=768:
                            pos=4
                        else:
                            continue
                    else:
                        distance=math.sqrt((rect_trans[0][0]-512)**2+(rect_trans[0][1]+offset-512)**2)
                        if distance<=256/3:
                            pos=5
                        elif distance>256/3 and distance<=256/3*2:
                            pos=6
                        elif distance>256/3*2 and distance<256:
                            pos=7
                        else:
                            continue
                    dist=[distance_to_line(k, b, point[0][0], point[0][1]) for point in points_trans]
                    dist=sorted(dist)
                    len_dist=len(dist)
                    lengthX=[h/w,outside_rect[0][1]-rect[0][1],abs((1536-b_fit)/k_fit-1024),max(rect_trans[1])]
                    diameterX=[h/w,outside_rect[0][1]-rect[0][1],abs((1536-b_fit)/k_fit-1024),sum(dist[int(0.25*len_dist):int(0.75*len_dist)])/(len_dist/2)]
                    if self.mode==1:
                        lengthY=self.pkl_square_screw_length.predict([lengthX,])[0]
                        diameterY=self.pkl_square_screw_diameter.predict([diameterX,])[0]
                    else:
                        lengthY=self.pkl_round_screw_length.predict([lengthX,])[0]
                        diameterY=self.pkl_round_screw_diameter.predict([diameterX,])[0]
                    screw_res.append([1,diameterY,lengthY,pos])
                    
            for gasket in gasket_res:
                tmp=np.finfo(np.float32).max
                for inside in inside_res:
                    distance=math.sqrt((gasket[2][0]-inside[2][0])**2+(gasket[2][1]-inside[2][1])**2)
                    if distance<tmp and distance<10:
                        tmp=distance
                        nearest_inside=inside
                if tmp<10:
                    combine_res.append([2,gasket[0],nearest_inside[0],gasket[1]])
                else:
                    combine_res.append([2,gasket[0],gasket[0]*0.6,gasket[1]])
            res="START\n"
            for screw in screw_res:
                res+="Goal_ID=1;Goal_A={};Goal_B={};Goal_C={}\n".format(screw[1],screw[2],screw[3])
            for combine in combine_res:
                res+="Goal_ID=2;Goal_A={};Goal_B={};Goal_C={}\n".format(combine[1],combine[2],combine[3])
            res+="END"
            try:
                self.socket.send(pack_data(DataType.RESULT, res))
            except Exception as e:
                print(e)
            if self.mode==1:
                filename=os.path.join(SAVE_DIR,"XJTU-SZRY-R1.txt")
            else:
                filename=os.path.join(SAVE_DIR,"XJTU-SZRY-R2.txt")
            with open(filename,"w") as file:
                file.write(res)
            self.table_result.setRowCount(0)
            for i in range(len(screw_res)):
                self.table_result.insertRow(i)
                self.table_result.setItem(i,0,QtWidgets.QTableWidgetItem('1'))
                self.table_result.setItem(i,1,QtWidgets.QTableWidgetItem(str(screw_res[i][1])))
                self.table_result.setItem(i,2,QtWidgets.QTableWidgetItem(str(screw_res[i][2])))
                self.table_result.setItem(i,3,QtWidgets.QTableWidgetItem(str(screw_res[i][3])))
            for i in range(len(combine_res)):
                self.table_result.insertRow(i+len(screw_res))
                self.table_result.setItem(i+len(screw_res),0,QtWidgets.QTableWidgetItem('2'))
                self.table_result.setItem(i+len(screw_res),1,QtWidgets.QTableWidgetItem(str(combine_res[i][1])))
                self.table_result.setItem(i+len(screw_res),2,QtWidgets.QTableWidgetItem(str(combine_res[i][2])))
                self.table_result.setItem(i+len(screw_res),3,QtWidgets.QTableWidgetItem(str(combine_res[i][3]))) 
            self.table_result.update()
            img=self.model.draw_and_visualize(frame, boxes, segments)
            showImage = QtGui.QImage(img,img.shape[1],img.shape[0],QtGui.QImage.Format_RGB888)
            self.label2.setPixmap(QtGui.QPixmap.fromImage(showImage).scaled(320,240))
            self.label_status.setText("识别完成,耗时{}s".format(time.time()-time0))
            self.predict_running=False
        except Exception as e:
            print(e)
            self.label_status.setText("识别失败")
            self.predict_running=False

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    mywin = UI()
    mywin.setWindowTitle('XJTU-Robocup-工业检测')
    mywin.show()
    sys.exit(app.exec_())

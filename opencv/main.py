import argparse
import cv2
import numpy as np
import math
import os


class Head_Pose():
    def __init__(self, modelpath, confThreshold=0.8, iou_threshold=0.5):
        self.net = cv2.dnn.readNet(modelpath)
        hw = os.path.basename(modelpath).split('_')[-1].split('.')[0].split('x')
        self.input_height = int(hw[0])
        self.input_width = int(hw[1])
        self.confThreshold = confThreshold
        self.iou_threshold = iou_threshold

    def detect(self, frame):
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (self.input_width, self.input_height), [0, 0, 0], swapRB=True,
                                     crop=False)
        self.net.setInput(blob)

        # Inference
        results = self.net.forward(self.net.getUnconnectedOutLayersNames())[0].squeeze(axis=0)  ##去掉开头的batch维度
        ###每一行的shape: center x, center y, width, height, box_conf, score, pitch, yaw, roll

        scale_h, scale_w = frame.shape[0] / self.input_height, frame.shape[1] / self.input_width
        bboxes_wh, confidences, pitchyawroll = [], [], []
        for box in results:
            conf = box[4] * box[5]  # conf = obj_conf * cls_conf
            if conf>self.confThreshold:
                xmin = max((box[0] - box[2] * 0.5) * scale_w, 0)
                ymin = max((box[1] - box[3] * 0.5) * scale_h, 0)
                xmax = min((box[0] + box[2] * 0.5) * scale_w, frame.shape[1])
                ymax = min((box[1] + box[3] * 0.5) * scale_h, frame.shape[0])
                bboxes_wh.append([xmin, ymin, xmax-xmin, ymax-ymin])
                confidences.append(conf)
                pitchyawroll.append(box[6:])
        indices = cv2.dnn.NMSBoxes(bboxes_wh, confidences, self.confThreshold, self.iou_threshold)
        if not isinstance(indices, (list, tuple)):
            indices = indices.flatten()
        faceboxes = []
        for i in indices:
            faceboxes.append(
                {'xmin': int(bboxes_wh[i][0]), 'ymin': int(bboxes_wh[i][1]), 'xmax': int(bboxes_wh[i][0]+bboxes_wh[i][2]), 'ymax': int(bboxes_wh[i][1]+bboxes_wh[i][3]), 'score': float(confidences[i]),
                 'pitch': float((pitchyawroll[i][0]-0.5)*180), 'yaw': float((pitchyawroll[i][1]-0.5)*360), 'roll': float((pitchyawroll[i][2]-0.5)*180)})
        return faceboxes

    def drawPred(self, frame, faceboxes):
        height, width = frame.shape[:2]
        for i, box in enumerate(faceboxes):
            # Draw a bounding box.
            cv2.rectangle(frame, (box['xmin'], box['ymin']), (box['xmax'], box['ymax']), (0, 0, 255), thickness=2)
            if not (math.isnan(box['yaw']) or math.isnan(box['pitch']) or math.isnan(box['roll'])):
                pitch = box['pitch'] * np.pi / 180
                yaw = -(box['yaw'] * np.pi / 180)
                roll = box['roll'] * np.pi / 180
                tdx = (box['xmin'] + box['xmax']) / 2
                tdy = (box['ymin'] + box['ymax']) / 2
                size = abs(box['xmax'] - box['xmin']) // 3
                if tdx != None and tdy != None:
                    tdx = tdx
                    tdy = tdy
                else:
                    tdx = width / 2
                    tdy = height / 2
                # X-Axis pointing to right. drawn in red
                x1 = size * (math.cos(yaw) * math.cos(roll)) + tdx
                y1 = size * (math.cos(pitch) * math.sin(roll) + math.cos(roll) * math.sin(pitch) * math.sin(yaw)) + tdy
                # Y-Axis | drawn in green
                #        v
                x2 = size * (-math.cos(yaw) * math.sin(roll)) + tdx
                y2 = size * (math.cos(pitch) * math.cos(roll) - math.sin(pitch) * math.sin(yaw) * math.sin(roll)) + tdy
                # Z-Axis (out of the screen) drawn in blue
                x3 = size * (math.sin(yaw)) + tdx
                y3 = size * (-math.cos(yaw) * math.sin(pitch)) + tdy
                cv2.line(frame, (int(tdx), int(tdy)), (int(x1), int(y1)), (0, 0, 255), 2)
                cv2.line(frame, (int(tdx), int(tdy)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.line(frame, (int(tdx), int(tdy)), (int(x3), int(y3)), (255, 0, 0), 2)

        return frame


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgpath", type=str, default='images/2.jpg', help="image path")
    parser.add_argument("--confThreshold", default=0.5, type=float, help='class confidence threshlod')
    parser.add_argument("--iouThreshold", default=0.5, type=float, help='nms threshlod')
    parser.add_argument("--modelpath", type=str, default='resources_nopost/directmhp_cmu_m_512x640.onnx',
                        help="onnxmodel path")  ###opencv读取["directmhp_300wlp_m_finetune_HxW.onnx", "directmhp_300wlp_s_finetune_HxW.onnx", "directmhp_agora_m_HxW.onnx", "directmhp_agora_s_HxW.onnx", "directmhp_cmu_m_HxW.onnx", "directmhp_cmu_s_HxW.onnx"]这些文件出错
    args = parser.parse_args()

    mynet = Head_Pose(args.modelpath, confThreshold=args.confThreshold, iou_threshold=args.iouThreshold)
    srcimg = cv2.imread(args.imgpath)
    faceboxes = mynet.detect(srcimg)
    srcimg = mynet.drawPred(srcimg, faceboxes)

    # cv2.imwrite('opencv_result.jpg', srcimg)
    winName = 'Deep learning Head Pose Estimation in OpenCV'
    cv2.namedWindow(winName, cv2.WINDOW_NORMAL)
    cv2.imshow(winName, srcimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

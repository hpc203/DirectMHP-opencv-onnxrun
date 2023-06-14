import argparse
import cv2
import numpy as np
import onnxruntime as ort
import math

class Head_Pose():
    def __init__(self, modelpath, confThreshold=0.8):
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.session = ort.InferenceSession(modelpath, so)
        model_inputs = self.session.get_inputs()
        self.input_name = model_inputs[0].name
        self.input_shape = model_inputs[0].shape
        self.input_height = int(self.input_shape[2])
        self.input_width = int(self.input_shape[3])
        self.confThreshold = confThreshold

    def detect(self, frame):
        dstimg = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        dstimg = cv2.resize(dstimg, (self.input_width, self.input_height))
        dstimg = dstimg.astype(np.float32) / 255.0
        input_image = np.expand_dims(dstimg.transpose(2, 0, 1), axis=0)

        # Inference
        results = self.session.run(None, {self.input_name: input_image})[0]
        ###shape: [N, [batchno, classid, x1, y1, x2, y2, score, pitch, yaw, roll]]

        keep_idxs = results[:, 6] > self.confThreshold
        batchno_classid_x1y1x2y2_score_pitchyawroll_keep = results[keep_idxs]
        scale_h, scale_w = frame.shape[0] / self.input_height, frame.shape[1] / self.input_width
        faceboxes = []
        for box in batchno_classid_x1y1x2y2_score_pitchyawroll_keep:
            xmin = max(box[2]*scale_w, 0)
            ymin = max(box[3]*scale_h, 0)
            xmax = min(box[4]*scale_w, frame.shape[1])
            ymax = min(box[5]*scale_h, frame.shape[0])
            faceboxes.append({'xmin':int(xmin), 'ymin':int(ymin), 'xmax':int(xmax), 'ymax':int(ymax), 'score':float(box[6]), 'pitch':float(box[7]), 'yaw':float(box[8]), 'roll':float(box[9])})
        return faceboxes

    def drawPred(self, frame, faceboxes):
        height, width = frame.shape[:2]
        for i,box in enumerate(faceboxes):
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
    parser.add_argument("--confThreshold", default=0.5, type=float, help='class confidence')
    parser.add_argument("--modelpath", type=str, default='resources_withpost/directmhp_cmu_m_post_512x640.onnx', help="onnxmodel path")
    args = parser.parse_args()

    mynet = Head_Pose(args.modelpath, confThreshold=args.confThreshold)
    srcimg = cv2.imread(args.imgpath)
    faceboxes = mynet.detect(srcimg)
    srcimg = mynet.drawPred(srcimg, faceboxes)

    # cv2.imwrite('onnxrun_result.jpg', srcimg)
    winName = 'Deep learning Head Pose Estimation in ONNXRuntime'
    cv2.namedWindow(winName, cv2.WINDOW_NORMAL)
    cv2.imshow(winName, srcimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
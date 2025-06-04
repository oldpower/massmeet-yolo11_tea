import cv2
import numpy as np
import onnxruntime as ort
from typing import List, Tuple

class YOLOv11nONNX:
    def __init__(self, 
                 model_path: str,
                 input_size: Tuple[int, int] = (640, 640),
                 conf_thresh: float = 0.25,
                 iou_thresh: float = 0.45):
        """
        初始化YOLOv11n ONNX推理器
        
        参数:
            model_path: ONNX模型路径
            input_size: 模型输入尺寸 (宽, 高)
            conf_thresh: 置信度阈值
            iou_thresh: NMS的IOU阈值
        """
        # 初始化参数
        self.input_size = input_size
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        
        # 创建ONNX Runtime会话
        self.session = ort.InferenceSession(model_path, 
                                         providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        
        # 获取输入输出名称
        self.input_name = self.session.get_inputs()[0].name
        print(f"Model input: {self.input_name}, shape: {self.session.get_inputs()[0].shape}")
        print(f"Model output: {self.session.get_outputs()[0].name}, shape: {self.session.get_outputs()[0].shape}")

    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, float, Tuple[float, float]]:
        """
        图像预处理 (resize + letterbox + normalization)
        
        返回:
            blob: 预处理后的图像 (1, 3, H, W)
            scale: 缩放比例
            pad: 填充像素 (宽, 高)
        """
        # 原始尺寸
        orig_h, orig_w = image.shape[:2]
        input_w, input_h = self.input_size
        
        # 计算缩放比例和填充
        scale = min(input_w / orig_w, input_h / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        pad_x = (input_w - new_w) // 2
        pad_y = (input_h - new_h) // 2
        
        # Letterbox缩放
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        blob = np.zeros((input_h, input_w, 3), dtype=np.uint8)
        blob[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
        
        # 归一化并转置为CHW
        blob = blob.astype(np.float32) / 255.0  # 0-1归一化
        blob = np.transpose(blob, (2, 0, 1))  # HWC -> CHW
        blob = np.expand_dims(blob, 0)  # 添加batch维度
        
        return blob, scale, (pad_x, pad_y)

    def postprocess(self, 
                    outputs: np.ndarray,
                    original_shape: Tuple[int, int],
                    scale: float,
                    pad: Tuple[float, float]) -> Tuple[List[List[int]], List[float], List[int]]:
        """
        后处理: 解码输出 + NMS
        
        返回:
            boxes: 检测框 [x1, y1, x2, y2] (原始图像坐标系)
            scores: 置信度（obj_conf * cls_score）
            class_ids: 类别ID
        """
        orig_h, orig_w = original_shape
        pad_x, pad_y = pad

        # 调整维度: [1, 7, 8400] → [7, 8400]
        preds = np.squeeze(outputs, axis=0)

        # 提取各通道数据
        cx = preds[0]         # 归一化中心x
        cy = preds[1]         # 归一化中心y
        w = preds[2]          # 归一化宽度
        h = preds[3]          # 归一化高度
        obj_conf = preds[5]   # 物体存在置信度
        cls_score = preds[6]  # 类别置信度（假设已经是最大类别的得分）

        # 可选：如果有多个类别的置信度（比如是向量），则使用下面这一行代替上面一行
        # class_scores = preds[5:]  # [2, 8400] for 2 classes
        # cls_score = np.max(class_scores, axis=0)
        # class_ids = np.argmax(class_scores, axis=0)

        # 结合 obj_conf 和 cls_score 得到最终置信度
        final_confidences = obj_conf * cls_score

        # 应用置信度阈值过滤
        keep = final_confidences > self.conf_thresh
        cx = cx[keep]
        cy = cy[keep]
        w = w[keep]
        h = h[keep]
        final_confidences = final_confidences[keep]
        class_ids = cls_score[keep].astype(int)  # 假设输出是类别 ID

        if len(final_confidences) == 0:
            return [], [], []

        input_w, input_h = self.input_size

        # 转换为 xyxy 坐标
        x1 = (cx - w / 2) * input_w
        y1 = (cy - h / 2) * input_h
        x2 = (cx + w / 2) * input_w
        y2 = (cy + h / 2) * input_h

        # 逆变换到原始图像坐标系
        x1 = (x1 - pad_x) / scale
        y1 = (y1 - pad_y) / scale
        x2 = (x2 - pad_x) / scale
        y2 = (y2 - pad_y) / scale

        # 限制在图像范围内
        x1 = np.clip(x1, 0, orig_w).astype(int)
        y1 = np.clip(y1, 0, orig_h).astype(int)
        x2 = np.clip(x2, 0, orig_w).astype(int)
        y2 = np.clip(y2, 0, orig_h).astype(int)

        # 执行NMS
        boxes = np.column_stack((x1, y1, x2 - x1, y2 - y1))  # xywh
        indices = cv2.dnn.NMSBoxes(
            bboxes=boxes.tolist(),
            scores=final_confidences.tolist(),
            score_threshold=self.conf_thresh,
            nms_threshold=self.iou_thresh
        )

        if len(indices) == 0:
            return [], [], []

        indices = indices.flatten()
        filtered_boxes = np.column_stack((x1[indices], y1[indices], x2[indices], y2[indices])).tolist()
        filtered_scores = final_confidences[indices].tolist()
        filtered_class_ids = class_ids[indices].tolist()

        return filtered_boxes, filtered_scores, filtered_class_ids

    def detect(self, image: np.ndarray) -> Tuple[List[List[int]], List[float], List[int]]:
        """完整检测流程"""
        # 预处理
        blob, scale, pad = self.preprocess(image)
        
        # 推理
        outputs = self.session.run(None, {self.input_name: blob})[0]
        
        # 后处理
        return self.postprocess(outputs, image.shape[:2], scale, pad)

    @staticmethod
    def draw_detections(image: np.ndarray, 
                       boxes: List[List[int]], 
                       scores: List[float],
                       class_ids: List[int],
                       color: Tuple[int, int, int] = (0, 255, 0),
                       thickness: int = 2) -> np.ndarray:
        """绘制检测结果"""
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(image, f"{score:.2f}", (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness)
        return image

# 使用示例
if __name__ == "__main__":
    # 初始化检测器
    onnx_path_yolo = r'/home/quan/yolo11_tea/runs/detect/train2/weights/best.onnx'
    input_img = '/home/quan/yolo11_tea/data/tea3/images/val/1 (2).jpg'

    detector = YOLOv11nONNX(
        model_path=onnx_path_yolo,  # 替换为你的模型路径
        conf_thresh=0.01,
        iou_thresh=0.5
    )
    
    # 读取图像
    image = cv2.imread(input_img)
    print('image shape:',image.shape)
    
    # 执行检测
    boxes, scores, class_ids = detector.detect(image)
    
    # 打印结果
    print(f"Detected {len(boxes)} objects:")
    for i, (box, score) in enumerate(zip(boxes, scores)):
        print(f"  Object {i + 1}: Box={box}, Score={score:.2f}")
    
    # 可视化
    result_image = detector.draw_detections(image.copy(), boxes, scores, class_ids)
    cv2.imwrite("result.jpg",result_image)
    # cv2.imshow("Detection Results", cv2.resize(result_image,(600,300)))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

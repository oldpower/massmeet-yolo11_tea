import cv2
import numpy as np
import onnxruntime
import time

# 类别名称
CLASS_NAMES = ['tea1', 'tea2', 'tea3']

# 预处理函数
def preprocess(img_path, img_size=640):
    img0 = cv2.imread(img_path)
    h0, w0 = img0.shape[:2]

    # 缩放比例与padding
    scale = min(img_size / w0, img_size / h0)
    new_w, new_h = int(w0 * scale), int(h0 * scale)
    resized = cv2.resize(img0, (new_w, new_h))

    # 填充灰色背景
    pad_w, pad_h = img_size - new_w, img_size - new_h
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2

    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114,114,114))
    img = padded.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)[np.newaxis, :]  # [1, 3, H, W]

    return img, img0, scale, left, top

# 后处理函数（适配无objectness输出格式）
def postprocess(output, img0, scale, pad_w, pad_h, conf_thres=0.501, iou_thres=0.45):
    output = output.squeeze().transpose(1, 0)  # [8400, 7]
    boxes = output[:, :4]
    class_scores = output[:, 4:]

    class_ids = np.argmax(class_scores, axis=1)
    confidences = np.max(class_scores, axis=1)
    confidences = 1/(1 + np.exp(-confidences))

    mask = confidences > conf_thres
    boxes = boxes[mask]
    class_ids = class_ids[mask]
    confidences = confidences[mask]

    if len(boxes) == 0:
        return [], [], []

    # [cx, cy, w, h] -> [x1, y1, x2, y2]
    boxes[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    boxes[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
    boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

    # 去除padding，除以scale恢复原图坐标
    boxes[:, [0, 2]] -= pad_w
    boxes[:, [1, 3]] -= pad_h
    boxes /= scale

    # NMS
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), confidences.tolist(), conf_thres, iou_thres)
    if len(indices) == 0:
        return [], [], []

    indices = indices.flatten()
    return boxes[indices], confidences[indices], class_ids[indices]

# 可视化
def draw_results(img, boxes, confs, class_ids):
    for box, conf, cls_id in zip(boxes, confs, class_ids):
        x1, y1, x2, y2 = map(int, box)
        label = f"{CLASS_NAMES[cls_id]} {conf:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 4)
        cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 4)
    return img

# 主推理流程
def run_inference(img_path, model_path="models/best.onnx"):
    session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    img, img0, scale, pad_w, pad_h = preprocess(img_path)

    start = time.time()
    outputs = session.run([output_name], {input_name: img})[0]
    infer_time = (time.time() - start) * 1000

    boxes, confs, class_ids = postprocess(outputs, img0, scale, pad_w, pad_h)

    print(f"Inference time: {infer_time:.2f} ms")
    for box, conf, cls in zip(boxes, confs, class_ids):
        print(f"{CLASS_NAMES[cls]}: {conf:.2f} - box: {box}")

    result_img = draw_results(img0, boxes, confs, class_ids)
    cv2.imwrite("result.jpg", result_img)
    print("Saved: result.jpg")
    # cv2.imshow('result',result_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    # 初始化检测器
    onnx_path_yolo = r'/home/quan/yolo11_tea/runs/detect/train2/weights/last.onnx'
    input_img = '/home/quan/yolo11_tea/data/tea3/images/val/1 (161).jpg'
    run_inference(input_img,onnx_path_yolo)

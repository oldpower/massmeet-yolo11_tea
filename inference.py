from ultralytics import YOLO
import cv2
import numpy as np


def test():
    # 加载导出的 ONNX 模型
    onnx_path_yolo = r'./runs/detect/train/weights/best.onnx'
    input_img = './data/images/val/1 (194).jpg'
    onnx_model = YOLO(onnx_path_yolo)
    # 进行推理
    results = onnx_model(input_img)[0]
    annotated_img = results.plot()
    cv2.imwrite('result.jpg',annotated_img)

def show():
    img = cv2.imread('result.jpg')
    cv2.imshow('result',cv2.resize(img,(600,300)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    # show()

    # exit(0)
    ckpt_path_yolo = r'/home/quan/yolo11_tea/runs/detect/train2/weights/best.pt'
    input_img = '/home/quan/yolo11_tea/data/tea3/images/val/1 (2).jpg'
    
    model_yolo = YOLO(model= ckpt_path_yolo)  
    predictions = model_yolo.predict(source=input_img,
                      save=False,
                      show=False,
                      conf = 0.2)
    # 获取第一个结果（单张图片）
    results = predictions[0]

    # 输出基本信息
    print("检测结果：")
    print(results.boxes)
    for box in results.boxes:
        cls_id = int(box.cls)             # 类别 ID
        conf = float(box.conf)            # 置信度
        bbox = box.xyxy.tolist()[0]       # 边界框坐标 [x1, y1, x2, y2]
        label = results.names[cls_id]     # 类别标签名称

        print(f"类别: {label}, 置信度: {conf:.2f}, 位置: {bbox}")

    # 可视化结果
    im_array = results.plot()  # 绘制预测框和标签
    im_show = cv2.cvtColor(im_array, cv2.COLOR_RGB2BGR)  # 转换为 BGR 用于 OpenCV 显示


    # 保存图像（可选）
    output_path = './output_result.jpg'
    cv2.imwrite(output_path, im_show)
    print(f"\n检测图像已保存至：{output_path}")

    # # 显示图像（可选）
    # img = cv2.imread(output_path)
    # cv2.imshow('Detection Result', cv2.resize(img,(400,200)))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

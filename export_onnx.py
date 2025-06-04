from ultralytics import YOLO

def export_01():
    model = YOLO(model=r'/home/quan/yolo11_tea/runs/detect/train2/weights/best.pt')   
    model.export(format="onnx",name = "./models/best.onnx")

def export_02():
    model = YOLO(model=r'/home/quan/yolo11_tea/runs/detect/train2/weights/last.pt')  
    model.export(format="onnx", dynamic=False, simplify=False, optimize=False, opset=12, imgsz=640)


def export_03():
    # 1. 加载 PyTorch 模型
    modelpath = r'./runs/detect/train/weights/best.pt'
    model = torch.load(modelpath, map_location='cpu')['model'].float()  # ultralytics格式读取
    model.eval()

    # 2. 生成 dummy input
    dummy_input = torch.randn(1, 3, 640, 640)

    # 3. ONNX 导出
    torch.onnx.export(
        model,
        dummy_input,
        'best_custom.onnx',
        opset_version=12,
        input_names=['images'],
        output_names=['output0'],
        dynamic_axes={
            'images': {0: 'batch', 2: 'height', 3: 'width'},
            'output0': {0: 'batch'}
        },
        do_constant_folding=True,
        verbose=True
    )

    print("ONNX model exported successfully.")


if __name__ == "__main__":
    export_03()
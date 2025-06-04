from ultralytics import YOLO

# Load a model
# model = YOLO("yolo11n.yaml").load("yolo11n.pt")  # build from YAML and transfer weights
model = YOLO("./yolo11n.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data="./data.yaml",
                      epochs=200,
                      imgsz=640,
                      batch = 64,
                      device='cuda:0')

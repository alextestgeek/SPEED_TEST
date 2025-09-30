import time
import cv2
import torch
from ultralytics import YOLO


torch.cuda.set_device(0)
model = YOLO("models/yolo11m.pt").to("cuda")


warmup_img = cv2.imread("test_image.jpg")
warmup_img = cv2.resize(warmup_img, (100, 100))
_ = model.track(warmup_img, verbose=False)
torch.cuda.synchronize()


print("\n=== Test 1: bus.jpg ===")
img = cv2.imread("bus.jpg")
img = cv2.resize(img, (640, 640))
torch.cuda.synchronize()
start_time = time.perf_counter()
results1 = model.track(img, verbose=False)
torch.cuda.synchronize()
time1 = time.perf_counter() - start_time
print(f"Direct call: {time1:.4f} s")

torch.cuda.empty_cache()
torch.cuda.synchronize()

print("\n=== Test 2: тензор test_image.jpg ===")
img2 = cv2.imread("test_image.jpg")
img2 = cv2.resize(img2, (640, 640))
img_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
tensor = torch.from_numpy(img_rgb).to("cuda")
tensor = tensor.permute(2, 0, 1).contiguous().unsqueeze(0) / 255.0
torch.cuda.synchronize()
start_time = time.perf_counter()
results2 = model.track(tensor, verbose=False)
torch.cuda.synchronize()
time2 = time.perf_counter() - start_time
print(f"тензор: {time2:.4f} s")

torch.cuda.empty_cache()
torch.cuda.synchronize()

print("\n=== Test 3: torch.compile bus.jpg ===")
model.model = torch.compile(model.model, mode="max-autotune", fullgraph=False, dynamic=False)
torch.cuda.synchronize()
_ = model.track(img, verbose=False)
torch.cuda.synchronize()
start_time = time.perf_counter()
results3 = model.track(img, verbose=False)
torch.cuda.synchronize()
time3 = time.perf_counter() - start_time
print(f"compiled direct: {time3:.4f} s")

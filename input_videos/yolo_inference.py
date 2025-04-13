from ultralytics import YOLO

model = YOLO('/Users/bjorn/Documents/football_analyses/ models/best.pt')
results = model.predict('/Users/bjorn/Documents/football_analyses/input_videos/08fd33_4.mp4', save = True)
print(results[0])
print('======')
for box in results[0].boxes:
    print(box)

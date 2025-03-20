import pandas as pd
import os

# ✅ 파일 경로 설정
train_csv_path = "/data/lwi2765/repos/XAI/Video-CBM-two-stream/data/video_annotation/UCF101/train_.csv"
output_csv_path = os.path.join(os.path.dirname(train_csv_path), "train.csv")  # ✅ 변환된 CSV 저장 경로

# ✅ train.csv 로드 (클래스명 포함된 비디오 파일 경로)
df = pd.read_csv(train_csv_path, header=None, names=["video_name", "index"], sep="\s+")

# ✅ "_result" 추가하여 변환
df["video_name"] = df["video_name"] + "_result.mp4"

# ✅ 새로운 CSV 파일 저장
df.to_csv(output_csv_path, sep=" ", index=False, header=False)
print(f"✅ 변환된 train.csv 저장 완료: {output_csv_path}")
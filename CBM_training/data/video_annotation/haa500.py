import pandas as pd

# ✅ 원본 CSV 파일 경로
input_csv = "/data/jong980812/project/Video-CBM-two-stream/data/video_annotation/haa500_subset/val.csv"  # 원본 CSV 파일 (경로 수정 가능)

# ✅ 필터링할 클래스 리스트 (예: 'CPR'만 포함)
selected_classes = ["backflip", "baseball_swing","basketball_shoot","frog_jump","golf_swing"]  # 원하는 클래스 이름 입력

# ✅ CSV 파일 로드
df = pd.read_csv(input_csv, header=None, names=["video_name", "class_label"])

# ✅ 특정 클래스만 필터링
filtered_df = df[df["video_name"].str.startswith(tuple(selected_classes))]

# ✅ 새로운 CSV 파일로 저장
output_csv = "/data/jong980812/project/Video-CBM-two-stream/data/video_annotation/haa500_small/val.csv"
filtered_df.to_csv(output_csv, index=False, header=False)

print(f"✅ {selected_classes} 클래스만 포함된 CSV 저장 완료: {output_csv}")
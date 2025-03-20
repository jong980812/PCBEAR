import pandas as pd

# ✅ 파일 경로 설정
original_csv_path = "/data/lwi2765/repos/XAI/Video-CBM-two-stream/data/video_annotation/haa49/original/train.csv"  # 원본 CSV 경로
remove_csv_path = "/data/lwi2765/repos/XAI/easy_ViTPose/cvprw_final/haa49/cl_10_finch_0_4404/missing_annotations.csv"  # 제거할 파일 목록이 담긴 CSV 경로
output_csv_path = "/data/lwi2765/repos/XAI/Video-CBM-two-stream/data/video_annotation/haa49/clip_level_10/train.csv"  # 필터링된 결과 저장 경로

# ✅ CSV 파일 로드
df1 = pd.read_csv(original_csv_path, header=None, names=["video_name", "class_id"], sep="\s+")
df2 = pd.read_csv(remove_csv_path,  header=None, names=["video_name"], sep="\s+")

# ✅ 제거할 비디오 리스트 추출
remove_list = set(df2["video_name"])

# ✅ 제거 수행
df_filtered = df1[~df1["video_name"].isin(remove_list)]

# ✅ 결과 저장
df_filtered.to_csv(output_csv_path, index=False, header=False, sep=" ")

print(f"✅ 필터링 완료! 원본 {len(df1)}개 → 필터링 후 {len(df_filtered)}개 저장")
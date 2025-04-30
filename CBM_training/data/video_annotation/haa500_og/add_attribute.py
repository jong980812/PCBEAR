import pandas as pd
import json
import pickle
# ✅ 경로 설정 (train.csv & annotation.json)
train_csv_path = "/data/jong980812/project/Video-CBM-two-stream/data/video_annotation/haa500_subset/val.csv"
annotation_json_path = "/data/jong980812/project/Video-CBM-two-stream/data/video_annotation/haa500_subset/annotations.json"  # 기존 annotation.json
output_json_path = "/data/jong980812/project/Video-CBM-two-stream/data/video_annotation/haa500_subset/hard_label_val.json"  # 순서 맞춘 새로운 JSON
output_pkl_path = "/data/jong980812/project/Video-CBM-two-stream/data/video_annotation/haa500_subset/hard_label_val.pkl"  # 순서 맞춘 새로운 JSON

# ✅ train.csv 불러오기 (비디오 순서 유지)
df = pd.read_csv(train_csv_path, header=None, names=["video_name", "class_label"])
video_list = df["video_name"].tolist()  # 비디오 순서 리스트

# ✅ annotation.json 불러오기
with open(annotation_json_path, "r") as f:
    annotation_data = json.load(f)

# ✅ annotation 데이터를 딕셔너리로 변환 (video_name → attribute_label 매핑)
annotation_dict = {
    item["video_name"].replace("/local_datasets/haa500/video/", "").replace("_result",""): item["attribute_label"]
    for item in annotation_data
}

# ✅ train.csv 순서대로 새로운 JSON 생성
sorted_annotations = []
for video_name in video_list:
    if video_name in annotation_dict:  # annotation이 존재하는 경우만 추가
        sorted_annotations.append({
            "video_name": video_name,
            "attribute_label": annotation_dict[video_name]
        })
    else:
        print(f"⚠ WARNING: {video_name}에 대한 annotation 정보 없음!")

# ✅ 새로운 JSON 파일로 저장
with open(output_json_path, "w") as f:
    json.dump(sorted_annotations, f, indent=4)
    
    
with open(output_pkl_path, "wb") as f:
    pickle.dump(sorted_annotations, f)

print(f"✅ train.csv 순서대로 정렬된 JSON 저장 완료: {output_json_path}")
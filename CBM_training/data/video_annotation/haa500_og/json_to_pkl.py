import pickle
import json

with open('hard_label_train.json', "r", encoding="utf-8") as f:
    data = json.load(f)

# pkl 파일로 저장
with open("hard_label_train.pkl", "wb") as f:
    pickle.dump(data, f)
    
    
with open('hard_label_val.json', "r", encoding="utf-8") as f:
    data = json.load(f)

# pkl 파일로 저장
with open("hard_label_val.pkl", "wb") as f:
    pickle.dump(data, f)
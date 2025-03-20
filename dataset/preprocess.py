# input_file = "/data/lwi2765/repos/XAI/PCBEAR/dataset/UCF-101/val2.csv"   # 원본 파일
# output_file = "/data/lwi2765/repos/XAI/PCBEAR/dataset/UCF-101/val.csv" # 결과 파일

# with open(input_file, "r") as infile, open(output_file, "w") as outfile:
#     for line in infile:
#         parts = line.rsplit(" ", 1)  # 마지막 공백을 기준으로 분리
#         try:
#             new_number = str(int(float(parts[1])) - 1)  # 숫자로 변환 후 -1
#             outfile.write(f"{parts[0]} {new_number}\n")
#         except ValueError:
#             outfile.write(line)  # 숫자로 변환 불가능한 경우 원본 유지

# print(f"변경된 파일이 {output_file}에 저장되었습니다.")

# # 텍스트 파일을 읽기
# df = pd.read_csv("/data/lwi2765/repos/XAI/PCBEAR/dataset/UCF-101/trainlist01.txt", sep=" ", header=None)

# # 공백을 구분자로 사용하여 CSV 저장
# df.to_csv("/data/lwi2765/repos/XAI/PCBEAR/dataset/UCF-101/train.csv", index=False, header=False, sep=" ")
input_file = "/data/lwi2765/repos/XAI/PCBEAR/dataset/UCF-101/classInd.txt"   # 원본 파일
output_file = "/data/lwi2765/repos/XAI/PCBEAR/dataset/UCF-101/class_list.txt" # 결과 파일

with open(input_file, "r") as infile, open(output_file, "w") as outfile:
    for line in infile:
        parts = line.split(" ", 1)  # 첫 번째 공백을 기준으로 분리
        if len(parts) > 1:
            outfile.write(parts[1])  # 숫자 부분 제외하고 저장

print(f"변경된 파일이 {output_file}에 저장되었습니다.")
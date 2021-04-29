
with open("EvaluationGT.csv", mode='r') as file:
    data1 = file.readlines()
with open("EvaluatedDataJerem.txt", mode='r') as file:
    data2 = file.readlines()

score = 0
for i in range(len(data1)):
    if data1[i] == data2[i]:
        score +=1

print("score", score, "likelihood", score/len(data1))



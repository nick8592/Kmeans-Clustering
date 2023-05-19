from sklearn.metrics import precision_score, recall_score

val_labels = \
[5, 5, 2, 4, 7, 7, 6, 5, 1, 4,
 3, 3, 0, 6, 8, 7, 6, 0, 4, 3,
 2, 9, 3, 7, 2, 8, 7, 9, 2, 0, 
 5, 2, 0, 3, 5, 0, 4, 1, 9, 1,
 2, 0, 3, 2, 9, 1, 4, 6, 7, 2, 
 8, 4, 8, 8, 0, 3, 1, 5, 9, 8, 
 3, 9, 1, 6, 6, 1, 4, 8, 5, 8, 
 2, 3, 7, 2, 6, 9, 7, 0, 5, 9, 
 1, 7, 8, 8, 9, 0, 6, 4, 1, 6, 
 9, 5, 7, 6, 1, 3, 5, 4, 4, 0]

predicted_labels = \
[2, 4, 2, 5, 4, 5, 1, 2, 4, 5, 
 7, 2, 8, 7, 6, 4, 8, 2, 4, 2, 
 2, 7, 2, 2, 2, 5, 2, 4, 4, 5, 
 2, 2, 6, 2, 7, 5, 6, 2, 0, 2, 
 2, 5, 5, 0, 2, 5, 5, 4, 5, 7, 
 1, 6, 2, 5, 5, 4, 5, 4, 5, 4, 
 5, 5, 5, 2, 5, 5, 5, 2, 4, 2, 
 3, 5, 6, 4, 2, 6, 0, 1, 4, 7, 
 0, 1, 5, 2, 9, 1, 5, 5, 2, 2, 
 5, 2, 4, 2, 4, 5, 4, 5, 5, 0]

for item in range(0, 10):
    val = [1 if n == item else 0 for n in val_labels]
    pred = [1 if n == item else 0 for n in predicted_labels]

    print(val)
    print(pred)
    pre = precision_score(val, pred, average='binary')
    rec = recall_score(val, pred, average='binary')
    print(pre)
    print(rec)
    input()


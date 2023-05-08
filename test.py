from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, precision_score, recall_score

# y_true = ["cat", "ant", "cat", "cat", "ant", "bird"]
# y_pred = ["ant", "ant", "cat", "cat", "ant", "cat"]
# mcm = multilabel_confusion_matrix(y_true, y_pred, labels=["ant", "bird", "cat"])
# print(mcm)

def get_precision(cm_arr):
    '''
    Precision = TP / (TP + FP)
    '''
    FP = cm_arr[0][1]
    TP = cm_arr[1][1]
    precision = TP / (TP + FP)
    return precision

def get_recall(cm_arr):
    '''
    Recall = TP / (TP + FN)
    '''
    FN = cm_arr[1][0]
    TP = cm_arr[1][1]
    recall = TP / (TP + FN)
    return recall

y_true = [5, 5, 2, 4, 7, 7, 6, 5, 1, 4,
 3, 3, 0, 6, 8, 7, 6, 0, 4, 3,
 2, 9, 3, 7, 2, 8, 7, 9, 2, 0, 
 5, 2, 0, 3, 5, 0, 4, 1, 9, 1,
 2, 0, 3, 2, 9, 1, 4, 6, 7, 2, 
 8, 4, 8, 8, 0, 3, 1, 5, 9, 8, 
 3, 9, 1, 6, 6, 1, 4, 8, 5, 8, 
 2, 3, 7, 2, 6, 9, 7, 0, 5, 9, 
 1, 7, 8, 8, 9, 0, 6, 4, 1, 6, 
 9, 5, 7, 6, 1, 3, 5, 4, 4, 0]

y_pred = [2, 4, 2, 5, 4, 5, 1, 2, 4, 5, 
 7, 2, 8, 7, 6, 4, 8, 2, 4, 2, 
 2, 7, 2, 2, 2, 5, 2, 4, 4, 5, 
 2, 2, 6, 2, 7, 5, 6, 2, 2, 2, 
 2, 5, 5, 3, 2, 5, 5, 4, 5, 7, 
 1, 6, 2, 5, 5, 4, 5, 4, 5, 4, 
 5, 5, 5, 2, 5, 5, 5, 2, 4, 2, 
 3, 5, 6, 4, 2, 6, 2, 1, 4, 7, 
 2, 1, 5, 2, 5, 1, 5, 5, 2, 2, 
 5, 2, 4, 2, 4, 5, 4, 5, 5, 4]

cm = confusion_matrix(y_true, y_pred)
print(f"Confusion Matrix: \n{cm}")

mcm = multilabel_confusion_matrix(y_true, y_pred)
print(f"Multi Confusion Matrix: \n{mcm}")

pre = precision_score(y_true, y_pred, average='micro')
rec = recall_score(y_true, y_pred, average='micro')
print(f"Total Precision: {pre}")
print(f"Total Recall:    {rec}")

for i in range(mcm.shape[0]):
    precision = get_precision(mcm[i])
    recall = get_recall(mcm[i])
    print(f"{i}. Precision: {precision}")
    print(f"{i}. Recall:    {recall}")

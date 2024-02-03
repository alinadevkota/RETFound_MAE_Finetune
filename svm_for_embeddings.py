from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import os
import torch
from sklearn.metrics import classification_report

embedding_path = "./idrid_embedding/"
def get_dataset():
    train_path = os.path.join(embedding_path, "train")
    test_path = os.path.join(embedding_path, "test")

    labels = os.listdir(train_path)

    train_X = []
    train_Y = []
    test_X = []
    test_Y = []
    for label in labels:
        train_label_path = os.path.join(train_path, label)
        for file_name in os.listdir(train_label_path):
            file_path = os.path.join(train_label_path, file_name)
            datapoint_X = torch.load(file_path, map_location=torch.device('cpu')).detach().numpy()
            train_X.append(datapoint_X)
            train_Y.append(label)

        test_label_path = os.path.join(test_path, label)
        for file_name in os.listdir(test_label_path):
            file_path = os.path.join(test_label_path, file_name)
            datapoint_X = torch.load(file_path, map_location=torch.device('cpu')).detach().numpy()
            test_X.append(datapoint_X)
            test_Y.append(label)

    return train_X, train_Y, test_X, test_Y
        
    

train_X, train_Y, test_X, test_Y = get_dataset()


# clf = svm.SVC(decision_function_shape='ovo')
clf = RandomForestClassifier(max_depth=2)


clf.fit(train_X, train_Y)
pred_Y = clf.predict(train_X)

print(classification_report(train_Y, pred_Y))

test_pred_Y = clf.predict(test_X)
print(classification_report(test_Y, test_pred_Y))
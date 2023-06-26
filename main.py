import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn import model_selection
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from keras.utils import to_categorical
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

data = pd.read_csv("train.csv")
data = data.iloc[1:]
data.reset_index(drop=True, inplace=True)

data = data.dropna()

X = data.drop('price_range', axis=1)
y = data['price_range']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

label_encoder = LabelEncoder()
y_test = label_encoder.fit_transform(y_test)
y_train = label_encoder.fit_transform(y_train)

y_train_classes = to_categorical(y_train, num_classes=4)
y_test_classes = to_categorical(y_test, num_classes=4)

kfold = model_selection.KFold(n_splits=5)

grid_models = []

param_grid = {
    'C': [0.001, 0.01, 1, 10, 100, 1000, 10000, 100000, 1000000]
}
softmax = LogisticRegression(multi_class='multinomial', solver='lbfgs', penalty='l2')
grid_models.append(('SoftMax',
                    GridSearchCV(estimator=softmax,
                                 param_grid=param_grid,
                                 cv=kfold,
                                 n_jobs=3)))

param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000],
    'gamma': [0.000001, 0.00001, 0.0001, 0.001, 0.1, 1, 10, 100]
}
svc = SVC(kernel='linear')
grid_models.append(('SVC linear',
                    GridSearchCV(estimator=svc,
                                 param_grid=param_grid,
                                 cv=kfold,
                                 n_jobs=3)))

param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10],
    'gamma': [0.001, 0.1, 1, 10, 100]
}
svc = SVC(kernel='poly')
grid_models.append(('SVC poly',
                    GridSearchCV(estimator=svc,
                                 param_grid=param_grid,
                                 cv=kfold,
                                 n_jobs=3)))

param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'gamma': [0.1, 1, 10, 100, 1000]
}
svc = SVC(kernel='rbf')
grid_models.append(('SVC rbf',
                    GridSearchCV(estimator=svc,
                                 param_grid=param_grid,
                                 cv=kfold,
                                 n_jobs=3)))

param_grid = {
    'n_estimators': [2000],
    'max_samples': [1000],
    'max_features': [20]
}
bagging_classifier = BaggingClassifier(DecisionTreeClassifier(), bootstrap=True)
grid_models.append(('Bagging',
                    GridSearchCV(estimator=bagging_classifier,
                                 param_grid=param_grid,
                                 cv=kfold,
                                 n_jobs=3)))

param_grid = {
    'n_estimators': [800],
    'max_samples': [500],
    'max_features': [20]
}
bagging_classifier = BaggingClassifier(DecisionTreeClassifier(), bootstrap=False)
grid_models.append(('Pasting',
                    GridSearchCV(estimator=bagging_classifier,
                                 param_grid=param_grid,
                                 cv=kfold,
                                 n_jobs=3)))

param_grid = {
    'n_estimators': [500],
    'min_samples_split': [3]
}
rf_classifier = RandomForestClassifier(max_depth=None)
grid_models.append(('Random Forest',
                    GridSearchCV(estimator=rf_classifier,
                                 param_grid=param_grid,
                                 cv=kfold,
                                 n_jobs=3)))

param_grid = {
    'criterion': ['entropy'],
    'max_depth': [None],
    'min_samples_split': [2],
    'min_samples_leaf': [2],
}
tree_classifier = DecisionTreeClassifier()
grid_models.append(('Decision Tree',
                    GridSearchCV(estimator=tree_classifier,
                                 param_grid=param_grid,
                                 cv=kfold,
                                 n_jobs=3)))


param_grid = {
    'n_estimators': [100],
    'learning_rate': [1.0]
}
ada_classifier = AdaBoostClassifier()
grid_models.append(('AdaBoost',
                    GridSearchCV(estimator=ada_classifier,
                                 param_grid=param_grid,
                                 cv=kfold,
                                 n_jobs=3)))

param_grid = {
    'n_estimators': [100],
    'learning_rate': [1.0]
}
gradient_classifier = GradientBoostingClassifier()
grid_models.append(('GradientBoost',
                    GridSearchCV(estimator=gradient_classifier,
                                 param_grid=param_grid,
                                 cv=kfold,
                                 n_jobs=3)))

model_scores = {}
model_names = []
model_precisions = []
model_recalls = []
model_f1s = []
model_accuracies = []

for name, model in grid_models:
    model.fit(X_train_scaled, y_train)
    best_estimator = model.best_estimator_
    y_pred = best_estimator.predict(X_test_scaled)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    accuracy = accuracy_score(y_test, y_pred)

    model_scores[name] = {
        'precision_score': precision,
        'recall_score': recall,
        'f1_score': f1,
        'accuracy_score': accuracy
    }

    model_names.append(name)
    model_precisions.append(precision)
    model_recalls.append(recall)
    model_f1s.append(f1)
    model_accuracies.append(accuracy)

df = pd.DataFrame.from_dict(model_scores, orient='index')
print(df)

plt.figure(figsize=(16, 6))
plt.bar(model_names, model_precisions)
for i in range(len(model_names)):
    plt.text(x=i, y=model_precisions[i]/2, s="{:.4f}".format(model_precisions[i]), ha='center')
plt.xlabel('Models')
plt.ylabel('Precision')
plt.title('Precision comparison')
plt.show()

plt.figure(figsize=(16, 6))
plt.bar(model_names, model_recalls)
for i in range(len(model_names)):
    plt.text(x=i, y=model_recalls[i]/2, s="{:.4f}".format(model_recalls[i]), ha='center')
plt.xlabel('Models')
plt.ylabel('Recall')
plt.title('Recall comparison')
plt.show()

plt.figure(figsize=(16, 6))
plt.bar(model_names, model_f1s)
for i in range(len(model_names)):
    plt.text(x=i, y=model_f1s[i]/2, s="{:.4f}".format(model_f1s[i]), ha='center')
plt.xlabel('Models')
plt.ylabel('F1')
plt.title('F1 comparison')
plt.show()

plt.figure(figsize=(16, 6))
plt.bar(model_names, model_accuracies)
for i in range(len(model_names)):
    plt.text(x=i, y=model_accuracies[i]/2, s="{:.4f}".format(model_accuracies[i]), ha='center')
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Accuracy comparison')
plt.show()

model_names = []
model_accuracy = []

model = Sequential()
model.add(Dense(256, activation="elu", input_shape=(X_train_scaled.shape[1],), kernel_initializer='he_uniform',
                use_bias=False))
model.add(BatchNormalization())
model.add(Dropout(0.6))
model.add(Dense(64, activation="elu", kernel_initializer='he_uniform', use_bias=False))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Dense(16, activation="elu", kernel_initializer='he_uniform', use_bias=False))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Dense(4, activation="softmax", kernel_initializer='he_uniform'))

early_stopping = EarlyStopping(monitor='val_loss', patience=5)

model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
model.fit(X_train_scaled, y_train_classes, batch_size=32, epochs=100, validation_data=(X_test_scaled, y_test_classes),
          callbacks=[early_stopping])

loss, accuracy = model.evaluate(X_test_scaled, y_test_classes)
model_names.append('elu')
model_accuracy.append(accuracy)

model = Sequential()
model.add(Dense(256, activation="relu", input_shape=(X_train_scaled.shape[1],), kernel_initializer='he_uniform',
                use_bias=False))
model.add(BatchNormalization())
model.add(Dropout(0.6))
model.add(Dense(64, activation="relu", kernel_initializer='he_uniform', use_bias=False))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Dense(16, activation="relu", kernel_initializer='he_uniform', use_bias=False))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Dense(4, activation="softmax", kernel_initializer='he_uniform'))

early_stopping = EarlyStopping(monitor='val_loss', patience=5)

model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
model.fit(X_train_scaled, y_train_classes, batch_size=32, epochs=65, validation_data=(X_test_scaled, y_test_classes),
          callbacks=[early_stopping])

loss, accuracy = model.evaluate(X_test_scaled, y_test_classes)
model_names.append('relu')
model_accuracy.append(accuracy)


model = Sequential()
model.add(Dense(256, input_shape=(X_train_scaled.shape[1],), kernel_initializer='he_uniform',
                use_bias=False))
model.add(LeakyReLU(alpha=0.2))
model.add(BatchNormalization())
model.add(Dropout(0.6))
model.add(Dense(64, kernel_initializer='he_uniform', use_bias=False))
model.add(LeakyReLU(alpha=0.2))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Dense(16, kernel_initializer='he_uniform', use_bias=False))
model.add(LeakyReLU(alpha=0.2))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Dense(4, activation="sigmoid", kernel_initializer='he_uniform'))

early_stopping = EarlyStopping(monitor='val_loss', patience=5)

model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
model.fit(X_train_scaled, y_train_classes, batch_size=32, epochs=40, validation_data=(X_test_scaled, y_test_classes),
          callbacks=[early_stopping])

loss, accuracy = model.evaluate(X_test_scaled, y_test_classes)
model_names.append('LeakyReLU')
model_accuracy.append(accuracy)

model = Sequential()
model.add(Dense(256, activation="sigmoid", input_shape=(X_train_scaled.shape[1],), kernel_initializer='he_uniform',
                use_bias=False))
model.add(BatchNormalization())
model.add(Dropout(0.6))
model.add(Dense(64, activation="sigmoid", kernel_initializer='he_uniform', use_bias=False))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Dense(16, activation="sigmoid", kernel_initializer='he_uniform', use_bias=False))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Dense(4, activation="sigmoid", kernel_initializer='he_uniform'))

early_stopping = EarlyStopping(monitor='val_loss', patience=5)

model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
model.fit(X_train_scaled, y_train_classes, batch_size=32, epochs=200, validation_data=(X_test_scaled, y_test_classes),
          callbacks=[early_stopping])

loss, accuracy = model.evaluate(X_test_scaled, y_test_classes)
model_names.append('sigmoid')
model_accuracy.append(accuracy)

model = Sequential()
model.add(Dense(256, activation="tanh", input_shape=(X_train_scaled.shape[1],), kernel_initializer='he_uniform',
                use_bias=False))
model.add(BatchNormalization())
model.add(Dropout(0.6))
model.add(Dense(64, activation="tanh", kernel_initializer='he_uniform', use_bias=False))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Dense(16, activation="tanh", kernel_initializer='he_uniform', use_bias=False))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Dense(4, activation="softmax", kernel_initializer='he_uniform'))

early_stopping = EarlyStopping(monitor='val_loss', patience=5)

model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
model.fit(X_train_scaled, y_train_classes, batch_size=32, epochs=100, validation_data=(X_test_scaled, y_test_classes),
          callbacks=[early_stopping])

loss, accuracy = model.evaluate(X_test_scaled, y_test_classes)
model_names.append('tanh')
model_accuracy.append(accuracy)

plt.figure(figsize=(10, 6))
plt.bar(model_names, model_accuracy)
for i in range(len(model_names)):
    plt.text(x=i, y=model_accuracy[i]/2, s="{:.4f}".format(model_accuracy[i]), ha='center')
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Accuracy comparison')
plt.show()
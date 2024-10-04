from preprocessing import load_and_preprocess_data
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score


X_train, X_val, X_test, y_train, y_val, y_test = load_and_preprocess_data(
    './archive/emnist-letters-train.csv', './archive/emnist-letters-test.csv'
)

# Log regression 
# need to comment the one-hot incoding part, reg modle need 1d array 
# log_reg_model = LogisticRegression(max_iter=1000)
# log_reg_model.fit(X_train, y_train)
# y_val_pred = log_reg_model.predict(X_val)
# print("Logistic Regression Validation Accuracy:", accuracy_score(y_val, y_val_pred))
# print("Classification Report:\n", classification_report(y_val, y_val_pred))

# Random forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_val_pred = rf_model.predict(X_val)
print("Random Forest Validation Accuracy:", accuracy_score(y_val, y_val_pred))
print("Classification Report:\n", classification_report(y_val, y_val_pred))

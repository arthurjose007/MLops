
#Modified
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
import mlflow

iris = load_iris()

data = pd.DataFrame(iris.data, columns=iris.feature_names)
data["species"] = iris.target

# Save dataset for DVC later
data.to_csv("iris.csv", index=False)

X = data.drop("species", axis=1)
y = data["species"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)



with mlflow.start_run():
    model = LogisticRegression(max_iter=50)
    model.fit(X_train, y_train)


mlflow.log_param("model", "LogisticRegression")
mlflow.log_param("max_iter", 100)  
mlflow.log_metric("accuracy", model.score(X_train, y_train))


joblib.dump(model, "model.pkl")

print("Model trained and saved successfully")

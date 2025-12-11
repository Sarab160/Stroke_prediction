import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score,recall_score,f1_score,confusion_matrix
from imblearn.over_sampling import SMOTE 


df=pd.read_csv("healthcare-dataset-stroke-data.csv")

print(df.head(1))
print(df.info())
df["bmi"]=df["bmi"].fillna(df["bmi"].mean())

print(df.describe())


print(df.info())

x=df[["age","hypertension","heart_disease","avg_glucose_level","bmi"]]
y=df["stroke"]

feature=df[["gender","ever_married","work_type","Residence_type","smoking_status"]]
le=OneHotEncoder(sparse_output=False,drop="first")
en=le.fit_transform(feature)
en_col=le.get_feature_names_out(feature.columns)
encode_data=pd.DataFrame(en,columns=en_col)

X=pd.concat([x,encode_data],axis=1)

sm=SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)



x_train,x_test,y_train,y_test=train_test_split(X_res,y_res,test_size=0.2,random_state=42)

dtc=KNeighborsClassifier(n_neighbors=5)

dtc.fit(x_train,y_train)

print("Test Score: ",dtc.score(x_test,y_test))
print("Train Score: ",dtc.score(x_train,y_train))

print("Precisoin score: ",precision_score(y_test,dtc.predict(x_test),average="macro"))
print("Recall score: ",recall_score(y_test,dtc.predict(x_test),average="macro"))
print("F1 Score: ",f1_score(y_test,dtc.predict(x_test),average="macro"))

score=confusion_matrix(y_test,dtc.predict(x_test))
sns.heatmap(score, annot=True, fmt="d", cmap="Blues")
plt.show()


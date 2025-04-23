import pandas as pd
from tkinter import *
from tkinter import messagebox
from tkinter import ttk
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import joblib


df = pd.read_csv("content_recommendation_data.csv")

le_target = LabelEncoder()


df["topic"] = le_target.fit_transform(df["topic"])

X = df[["scroll_depth", "time_spent"]]
y = df["topic"]


dt_model = KMeans(n_clusters=3,random_state=42)
dt_model.fit(X, y)
joblib.dump(dt_model, "expiry_model_dt.pkl")


label_map = dict(zip(le_target.transform(le_target.classes_), le_target.classes_))


def predict_type():
    try:
        scroll_depth = scroll_depth_entry.get()
        sales = float(time_spent_entry.get())
        
        
        prediction = dt_model.predict([[scroll_depth, sales]])[0]
        Topic = label_map[prediction]

        messagebox.showinfo("Prediction Result", f"Recommended Topic: {Topic}")
    except Exception as e:
        messagebox.showerror("Error", str(e))


root = Tk()
root.title("Smart Inventory Expiry Predictor")
root.geometry("400x350")

Label(root, text="Scroll_Depth(1-10):").pack(pady=5)
scroll_depth_entry = Entry(root)
scroll_depth_entry.pack()

Label(root, text="Time_Spent(1-10):").pack(pady=5)
time_spent_entry = Entry(root)
time_spent_entry.pack()



Button(root, text="Recommended Topic", command=predict_type).pack(pady=20)

root.mainloop()   
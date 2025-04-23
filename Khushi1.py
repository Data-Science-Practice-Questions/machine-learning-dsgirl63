import pandas as pd
from tkinter import *
from tkinter import messagebox
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import joblib

tips = {
    0: "No success",
    1: "Successfull."
}



df = pd.read_csv("film_script_success_data.csv")
le_product = LabelEncoder()
le_target = LabelEncoder()

df["genre"] = le_product.fit_transform(df["genre"])
df["genre"] = le_target.fit_transform(df["genre"])

X = df[["genre", "sentiment_score", "keyword_count", "dialogue_length"]]
y = df["success"]


dt_model = LogisticRegression(random_state=42)
dt_model.fit(X, y)
joblib.dump(dt_model, "expiry_model_dt.pkl")


label_map = dict(zip(le_target.transform(le_target.classes_), le_target.classes_))


def predict_expiry():
    try:
        ptype = movie_type_entry.get()
        sales = float(sentiment_score_entry.get())
        temp = float(keyword_count_entry.get())
        humidity = float(dialogue_length_entry.get())

        ptype_encoded = le_product.transform([ptype])[0]
        prediction = dt_model.predict([[ptype_encoded, sales, temp, humidity]])[0]
        risk = label_map[prediction]

        tip = tips.get(prediction, "Tip not available.")

        messagebox.showinfo("Prediction Result", f"Predicted Expiry Risk: {risk}\n {tip}")
    except Exception as e:
        messagebox.showerror("Error", str(e))


root = Tk()
root.title("Film Script Success Predictor")
root.geometry("400x350")

Label(root, text="Product Type (Horror, Sci-Fi, Comedy, Drama):").pack(pady=5)
movie_type_entry = Entry(root)
movie_type_entry.pack()

Label(root, text="Sentiment_score:").pack(pady=5)
sentiment_score_entry = Entry(root)
sentiment_score_entry.pack()

Label(root, text="keyword_count:").pack(pady=5)
keyword_count_entry = Entry(root)
keyword_count_entry.pack()

Label(root, text="dialogue_length:").pack(pady=5)
dialogue_length_entry = Entry(root)
dialogue_length_entry.pack()

Button(root, text="Predict Success Rate + tip", command=predict_expiry).pack(pady=20)

root.mainloop()   
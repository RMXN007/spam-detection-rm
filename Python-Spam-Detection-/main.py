import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report


# Step 1: Load the dataset
df = pd.read_csv("spam.csv")

df = df.dropna(subset=["label", "message"])
df["label"] = df["label"].str.strip().str.lower()
df["message"] = df["message"].astype(str)

df = df[df["label"].isin(["ham", "spam"])]
df = df.drop_duplicates()

print("Dataset loaded successfully!")
print("Total messages:", len(df))
print("Spam messages:", (df["label"] == "spam").sum())
print("Genuine messages:", (df["label"] == "ham").sum())


# Step 2: Separate messages and labels
X = df["message"]
y = df["label"]


# Step 3: Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Step 4: Create the machine learning pipeline
model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2)
        )
    ),
    ("classifier", MultinomialNB())
])


# Step 5: Train the model
print("\nTraining the model...")
model.fit(X_train, y_train)
print("Model trained successfully!")


# Step 6: Evaluate the model
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        labels=["ham", "spam"],
        zero_division=0
    )
)


# Step 7: Predict new messages
while True:
    print("\n========== SPAM DETECTOR ==========")
    print("1. Check a message")
    print("2. Exit")

    choice = input("Enter your choice: ").strip()

    if choice == "1":
        message = input("Enter your message: ").strip()

        if not message:
            print("Please enter a message.")
            continue

        prediction = model.predict([message])[0]
        probabilities = model.predict_proba([message])[0]
        classes = list(model.classes_) if hasattr(
            model, "classes_"
        ) else list(
            model.named_steps["classifier"].classes_
        )

        confidence = probabilities[classes.index(prediction)] * 100

        if prediction == "spam":
            print("\nResult: SPAM MESSAGE")
            print("Action: Flag this message.")
        else:
            print("\nResult: GENUINE MESSAGE (HAM)")
            print("Action: Keep this message.")

        print(f"Model confidence: {confidence:.2f}%")

    elif choice == "2":
        print("Thank you for using Spam Detector!")
        break

    else:
        print("Invalid choice. Enter 1 or 2.")

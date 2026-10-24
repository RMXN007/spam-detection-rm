
import pandas as pd

df = pd.read_csv(
    "SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"],
    encoding="utf-8"
)

df = df.dropna()
df = df.drop_duplicates()

df.to_csv("spam.csv", index=False)

print("Dataset converted successfully!")
print("Total messages:", len(df))
print(df.head())
print("\nMessage counts:")
print(df["label"].value_counts())

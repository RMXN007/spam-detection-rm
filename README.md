# SMS Spam Detection

A command-line machine learning application that classifies SMS messages as
**spam** or **ham** (legitimate, non-spam messages).

The project demonstrates a complete text-classification workflow:

1. Load and clean a labelled SMS dataset.
2. Split the data into training and testing sets.
3. Convert message text into numerical TF-IDF features.
4. Train a Multinomial Naive Bayes classifier.
5. Measure the model on unseen test messages.
6. Classify messages entered interactively by the user.

## How the model works

The model is implemented as a scikit-learn `Pipeline`:

```text
SMS text
   |
   v
TF-IDF vectorizer
   |
   v
Multinomial Naive Bayes classifier
   |
   v
ham or spam + confidence
```

### 1. Data cleaning

`main.py` reads `spam.csv` and:

- removes rows without a label or message;
- normalizes labels to lowercase;
- keeps only the `ham` and `spam` classes;
- removes duplicate messages.

The data is split into 80% training data and 20% testing data. The split uses
`random_state=42` for repeatable results and `stratify=y` so both classes are
represented proportionally in each set.

### 2. TF-IDF feature extraction

Computers cannot train directly on raw sentences, so
`TfidfVectorizer` converts each message into numerical features. The vectorizer
lowercases text, removes English stop words, and considers both individual
words and two-word combinations (`ngram_range=(1, 2)`).

TF-IDF gives higher importance to terms that are useful for distinguishing
messages while reducing the importance of terms that occur frequently across
the dataset.

### 3. Classification

`MultinomialNB` (Multinomial Naive Bayes) learns the relationship between the
TF-IDF features and the `ham`/`spam` labels. It is a common, efficient
baseline for text classification.

The program reports accuracy and a classification report containing precision,
recall, and F1-score for both classes. It then provides an interactive menu
for testing new messages. For each prediction, it displays the predicted class
and the model's probability-based confidence.

## Project structure

| File | Description |
| --- | --- |
| [`main.py`](./main.py) | Loads the prepared CSV, trains and evaluates the model, and runs the interactive detector. |
| [`prepare_data.py`](./prepare_data.py) | Converts the tab-separated source dataset into `spam.csv`. |
| [`SMSSpamCollection`](./SMSSpamCollection) | Original SMS dataset: one label and one message per tab-separated row. |
| [`spam.csv`](./spam.csv) | Prepared CSV dataset with `label` and `message` columns. |
| [`requirements.txt`](./requirements.txt) | Python dependencies required by the project. |

## Requirements

- Python 3.8 or newer
- `pandas`
- `scikit-learn`

## Installation

Open a terminal in the project directory and create an optional virtual
environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

## Running the project

The prepared `spam.csv` file is already included, so the application can be
started directly:

```bash
python main.py
```

The program first prints the dataset counts, trains the model, and displays
the evaluation metrics. It then opens this menu:

```text
========== SPAM DETECTOR ==========
1. Check a message
2. Exit
```

Choose `1`, enter an SMS message, and the application will print whether it is
spam or ham and show the confidence. Choose `2` to stop the program.

### Rebuilding the prepared dataset

If the original `SMSSpamCollection` file changes or `spam.csv` needs to be
regenerated, run:

```bash
python prepare_data.py
python main.py
```

`prepare_data.py` reads the original tab-separated file, assigns the columns
`label` and `message`, removes empty and duplicate rows, and writes the result
to `spam.csv`.

## Dataset format

The application expects `spam.csv` to contain these columns:

```csv
label,message
ham,"Your legitimate message here"
spam,"Your promotional or fraudulent message here"
```

Labels must be `ham` or `spam` (case and surrounding whitespace are cleaned
by `main.py`). The included dataset contains real SMS examples and is intended
for educational use.

## Example

After training, enter a message such as:

```text
Congratulations! You have won a free prize. Call now to claim it.
```

The application returns a result similar to:

```text
Result: SPAM MESSAGE
Action: Flag this message.
Model confidence: 98.42%
```

The exact confidence and prediction depend on the dataset and the trained
model. Confidence is the classifier's estimated probability, not a guarantee
that a message is spam or legitimate.

## Limitations

- This is an educational command-line classifier, not a production spam
  filter.
- Predictions depend on the vocabulary and patterns present in the training
  dataset. New types of spam may be misclassified.
- The model uses only message text; it does not inspect sender identity,
  links, attachments, or message metadata.
- Accuracy alone can be misleading when classes are imbalanced, so precision,
  recall, and F1-score should also be considered.
- The model is trained each time `main.py` runs and is not saved to disk.

## License and usage

No license is currently specified for this project. Check the source dataset's
terms before redistributing it or using the project outside coursework.

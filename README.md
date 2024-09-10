
# Sentiment Analysis Web App

This web application was built using **Streamlit** to perform sentiment analysis on tweets. The model predicts whether a tweet is positive, neutral, or negative.

![sampal news](https://github.com/user-attachments/assets/88ca7ef7-4580-4264-b143-87a99e4ba3bb)


## Features

- Simple UI to input a tweet for sentiment analysis.
- Pre-trained model that classifies tweets into one of three categories: **Positive**, **Neutral**, or **Negative**.
- The model has an accuracy of **77.06%**.

## How It Works

1. The user inputs a tweet into the provided text field.
2. The app cleans and preprocesses the text by:
   - Removing non-alphabetic characters.
   - Converting the text to lowercase.
   - Removing stopwords (common words that do not contribute to sentiment).
   - Lemmatizing words (converting them to their root form).
3. The pre-trained sentiment analysis model predicts the sentiment based on the processed tweet.
4. The result is displayed as either **Positive**, **Neutral**, or **Negative**.

## Model

The model used in this project was pre-trained and saved in a `.pkl` file. The project also uses a **CountVectorizer** to transform the input text into a format the model can process.

## Installation and Usage

1. Clone the repository:

```bash
git clone "https://github.com/More-Sushant/Sentiment-Analysis.git"
```

2. Navigate to the project directory:

```bash
cd Sentiment-Analysis-ML
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```

4. Place the pre-trained model files (`Model.pkl` and `cv.pkl`) in the `./models/` directory.

5. Run the Streamlit app:

```bash
streamlit run app.py
```

6. Open your browser and go to `http://localhost:8501` to access the app.

## Requirements

- Python 3.x
- Streamlit
- NLTK
- Scikit-learn
- Pickle

You can install the necessary dependencies using the `requirements.txt` file provided in the repository.

## Project Structure

```bash
.
├── app.py                  # Main Streamlit app script
├── models/
│   ├── Model.pkl           # Pre-trained sentiment analysis model
│   └── cv.pkl              # CountVectorizer used for text transformation
├── modules/                
│   └── module_sentiment.py # Python file for testing accuracy of different models
├── requirements.txt        # List of required Python packages
└── README.md               # Project documentation
```

## Acknowledgements

- The NLTK library is used for text preprocessing (lemmatization and stopword removal).
- Scikit-learn is used for building and training the model.
- Streamlit is used for the web interface.

## License

This project is licensed under the MIT License.

---

Upvote, Share and Follow for more⬆️🔝

---

Made with ❤️ by Sushant

# Grand Line Message Bounty Detector — Dataset Setup

## Dataset Source & Attribution
The dataset used in this project is the **UCI SMS Spam Collection**:
- **Source:** [UCI Machine Learning Repository - SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)
- **Creators & Authors:** Tiago A. Almeida and José María Gómez Hidalgo.

## Manual Dataset Download Instructions

1. Download the dataset zip file from the official UCI repository URL or direct link:
   `https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip`
2. Extract the downloaded zip archive.
3. Locate the file named `SMSSpamCollection` (it does not have a file extension).
4. Place `SMSSpamCollection` directly inside this `data/` directory so its exact path is:
   `data/SMSSpamCollection`

## Data Format

`SMSSpamCollection` is a tab-separated text file formatted as follows:
- **Column 1 (`label`):** Either `ham` (legitimate message) or `spam` (unwanted/fraudulent message).
- **Column 2 (`message`):** Raw plain-text string of the message.

Example entries:
```text
ham	Go until jurong point, crazy.. Available only in bugis n great world la e buffet...
spam	Free entry in 2 a wk compensatory entry to win FA Cup final tkts 21st May 2005. Text FA to 87121...
```

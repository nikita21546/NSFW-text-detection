### Project Title: NSFW Content Classifier

#### Project Description:
This project aims to create a robust machine learning model capable of identifying and filtering Not Safe For Work (NSFW) text content, primarily on social media platforms like Reddit. The primary objective is to ensure a safer online environment by effectively excluding NSFW content using various machine learning techniques.

#### Team Members:
- Ritisha Singh (2021089)
- Nikita Rajesh Verma (2021546)
- Jeremiah Malsawmkima Rokhum (2021533)
- Sanskar Ranjan (2021096)

#### Abstract:
The proliferation of NSFW content online poses significant risks, including mental health issues and workplace disruptions. This project develops four machine learning models—Naive Bayes, Logistic Regression, Support Vector Machines (SVM), and Convolutional Neural Networks (CNN)—to classify and filter NSFW text data. Each model is trained on a dataset from social media platforms like Reddit, aiming to create an effective classifier to enhance online safety.

#### Introduction:
With the vast amount of content hosted on the internet, ensuring a safe online environment is crucial. This project leverages machine learning techniques to develop an automated system for detecting NSFW content on Reddit, aiming to maintain a safe and appropriate online experience for users.

#### Literature Survey:
The literature survey includes studies on NSFW text identification using Transformer-based models and various classifiers like Naive Bayes and Stochastic Gradient Descent. These studies highlight the challenges and effectiveness of different models in identifying NSFW content.

#### Dataset:
The dataset comprises text comments with toxicity labels. The primary goal is to predict whether these comments contain toxic content. The dataset includes subtype attributes such as severe toxicity, obscene, threat, insult, identity attack, and sexual explicit, providing a comprehensive basis for training the models.

#### Data Pre-processing Techniques:
1. **Lower-casing**: Converting all text characters to lowercase for consistency.
2. **Removal of Special Characters & Contractions**: Removing noise from the text.
3. **Removal of Stopwords**: Excluding common words to focus on meaningful content.
4. **Tokenization**: Splitting text into individual tokens.
5. **Stemming**: Reducing words to their root form.

#### Methodology:
1. **Bag of Words (BoW)**: Represents documents as an unordered collection of words.
2. **Term Frequency-Inverse Document Frequency (TF-IDF)**: Evaluates the importance of a word in a document relative to a corpus.
3. **Logistic Regression**: Used for binary classification.
4. **Naive Bayes**: A probabilistic algorithm used for text categorization.
5. **Random Forest**: An ensemble algorithm that builds multiple decision trees.
6. **Support Vector Machine (SVM)**: Finds the optimal hyperplane that separates different classes.
7. **Convolutional Neural Network (CNN)**: Uses layers to automatically extract features from input data.

#### Results and Analysis:
The models were evaluated based on their accuracy and F1 scores. The SVM model demonstrated exceptional performance with an accuracy of 0.93675, followed by CNN with 0.9273. Random Forest, Logistic Regression (BoW and TF-IDF), and Naive Bayes also showed varying degrees of effectiveness.

#### Conclusion:
The project highlighted the challenges of NSFW detection, particularly the bias towards non-toxic comments in the dataset. Despite resource limitations, meaningful results were achieved, demonstrating the potential of machine learning in addressing NSFW content challenges. Future research should focus on addressing data bias and exploring advanced techniques to improve model accuracy.

#### References:
1. Classifying Reddit Posts with Natural Language Processing and Machine Learning - Medium
2. NSFW Text Identification - ResearchGate
3. Classification of Reddit Posts: Predicting “Not Safe For Work” Content - University of British Columbia
4. Jigsaw Unintended Bias in Toxicity Classification - Kaggle

For more detailed information regarding this project, please refer to the [Report](Report.pdf).

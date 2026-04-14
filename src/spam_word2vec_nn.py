import pandas as pd
import numpy as np
import tensorflow.keras as keras
from gensim.models import word2vec
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score



# Read the CSV file
mydata = pd.read_csv('spam_or_not_spam.csv')

# Keep only the emails
emails= mydata['email']

output=mydata['label']

email_split=[]

# For each email, split it into a list of words
for email in emails:
    email_split.append(str(email).split())

# Apply the Word2Vec model in order to transform words into vectors
model = word2vec.Word2Vec(email_split, min_count=1)


vectorized_emails=[]

# For each element of the email_split list (which is a list of words)
for x in email_split:
  # The list of vectors for each email
    vector_email  = []
   # For each word in the email
    for y in x:
      # Add the encoded vector of the word to vector_email
        vector_email.append(model.wv[y])
    # Add to the main list the list of vectors corresponding to the email
    vectorized_emails.append(vector_email)
  
final_emails=[]



# Add to final_emails the average of the vectors of the words that make up the email
# Since each email consists of many vectors, and we want one final vector per email,
# we take the average of the vectors of its words
for x in vectorized_emails:

    final_emails.append(np.average(x, axis=0))


# Transform the data into a suitable format so that it can be used
# as input to the neural network

# List of inputs
final_input=[]

# List of outputs
final_output=[]

# For each email that has been vectorized
for i in range(len(final_emails)):
    # If it is not NaN
    if not np.isnan(final_emails[i]).all():
        # Create a temporary list
        temp_list=[]
        # For each value in the email vector
        for x in final_emails[i]:
             # Add it to the temporary list
            temp_list.append(x)
        # Add the temporary list to the final input list
        final_input.append(temp_list)
        # Add the corresponding output label to the final output list
        final_output.append(output[i])

# final_input is a list of lists

# final_output is a list

# Convert the input data list into a 2D array so it can be fed into the neural network
final_array = np.array(final_input)

# Convert the output data list into a 1D array so it can be fed into the neural network
final_output = np.array(final_output)

X_train, X_test, y_train, y_test = train_test_split(final_array, final_output, test_size=0.25)

# Create the neural network model

# Define the topology

# Note: input_dim must be 100

model = Sequential()
model.add(Dense(12, input_dim=100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the neural network
model.fit(X_train, y_train, epochs=20, batch_size=100)

# Prediction from the neural network
y_pred=model.predict(X_test)


rounded_predictions = [int(round(x[0])) for x in y_pred]



precision = precision_score(y_test, rounded_predictions, average='binary')

print('Precision: %.3f' % precision)

recall = recall_score(y_test, rounded_predictions, average='binary')
print('Recall: %.3f' % recall)

score = f1_score(y_test, rounded_predictions, average='binary')
print('F-Measure: %.3f' % score)











  
    

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



#read of the csv file
mydata = pd.read_csv('spam_or_not_spam.csv')

#here we keep only the emails 
emails= mydata['email']

output=mydata['label']

email_split=[]

#for each email we do a split and we create a list of words
for email in emails:
    email_split.append(str(email).split())

#implementing word2vec model in order to transform the words into vectors 
model = word2vec.Word2Vec(email_split, min_count=1)


vectorized_emails=[]

#for each element of the list email_split(which is a word list)
for x in email_split:
   #the list with the vectors for each email
    vector_email  = []
    #for each word of the email 
    for y in x:
      #prosthetoume to kodikopoihmeno dianisma(poy proekypse apo tin texniki word embedings) sto vector_email
        vector_email.append(model.wv[y])
    #prosthetoume stin megali lista, tin lista twn dianismatwn poy antistoixei sto email    
    vectorized_emails.append(vector_email)
  
final_emails=[]




#prosthetoume ston final_emamils ton meso oro twn dianismatwn twn leksewn pou apoteloun to email
#(epeidi kathe email apoteleitai apo polla dianismata, epeidi theloume na exoume telika ena dianisma 
#ana email, pairnoume ton meso oro twn dianismatwn twn leksewn pou ton apoteloun
for x in vectorized_emails:

    final_emails.append(np.average(x, axis=0))


#ginetai metasximatismos twn dedomenwn se katallili morfi etsi wste na mporoun na eisaxthoun
#sto nevroniko diktyo

#lista me tis eisidous
final_input=[]

#lista me tis eksodous 
final_output=[]

#gia kathe email(to opoio exei ypostei dianismatopoihsi)
for i in range(len(final_emails)):
    #an den einai nAN
    if not np.isnan(final_emails[i]).all():
        #dimiourgoume mia prosoroni lista
        temp_list=[]
        #gia kathe dianisma pou einai mesa sto email
        for x in final_emails[i]:
            #to prosthetoume stin prosorini lista
            temp_list.append(x)
        #prosthetoume tin prosorini lista stin teliki (gia tis eisodous)
        final_input.append(temp_list)
        #prostetoume tin eksodo stin teliki lista (gia tis eksodous )
        final_output.append(output[i])

#ara to final_input einai mia lista apo listes 

#kai to final_output einai mia lista

#metatrepoume tin lista (me ta dedomena eisodou ) se 2d array etsi wste na mporesei na eisaxthei sto nevroniko diktyo
final_array = np.array(final_input)

#metatrpoume tin lista (me ta dedomena eksodou) se 1d array etsi wste na mporei na eisaxthei sto nevroniko diktyo
final_output = np.array(final_output)

X_train, X_test, y_train, y_test = train_test_split(final_array, final_output, test_size=0.25)

#dimiourgia montelou tou nevronikou diktyou

#dimiourgia tis topologias 

#prosoxu : to input_dim prepei na einai 100
model = Sequential()
model.add(Dense(12, input_dim=100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#edo ginetai i ekpaideysi tou nevronikou diktoty

model.fit(X_train, y_train, epochs=20, batch_size=100)

#prediction apo to nevroniko diktyo

y_pred=model.predict(X_test)


rounded_predictions = [int(round(x[0])) for x in y_pred]



precision = precision_score(y_test, rounded_predictions, average='binary')

print('Precision: %.3f' % precision)

recall = recall_score(y_test, rounded_predictions, average='binary')
print('Recall: %.3f' % recall)

score = f1_score(y_test, rounded_predictions, average='binary')
print('F-Measure: %.3f' % score)











  
    

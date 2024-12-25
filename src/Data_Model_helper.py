import tensorflow.keras as keras
from keras.models import Model
import pandas as pd
import tensorflow as tf
import numpy as np
from numpy import zeros
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Input, Dropout, Embedding, Bidirectional, LSTM, GRU, Flatten, LayerNormalization, BatchNormalization, AdditiveAttention, MultiHeadAttention, GlobalAveragePooling1D #change added last three
from tensorflow.keras.preprocessing.text import Tokenizer, one_hot
from tensorflow.keras.layers import Conv1D, MaxPooling1D #change
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers

from nltk.tokenize import TweetTokenizer
from transformers import AutoTokenizer, AutoModel
from transformers import DistilBertTokenizer, TFDistilBertModel
from transformers import BertTokenizer, TFBertModel, TFGPT2Model, GPT2Tokenizer

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, roc_curve, auc #change added last two
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight

from gensim.models.doc2vec import Doc2Vec, TaggedDocument

import seaborn as sn
import matplotlib.pyplot as plt


import re

###############################################################################################################
# These are model helper
#################################################################################################################

#split train val test
def split_data(cleantweet):
  trainX, tempX = train_test_split(cleantweet, test_size=0.4, random_state=42)
  valX, testX = train_test_split(tempX, test_size=0.5, random_state=42)
  return trainX, valX, testX

#extract tweet and y
def extract_tweet_and_y(raw_data_df):
  tweet, target = raw_data_df['clean_tweet'], raw_data_df['class']
  return tweet, target

#tokenize and vectorize input using keras tokenizer
def keras_tokenizer(tweet_train, tweet_val, tweet_test, maxnumwords):
  # maxnumwords = 2000
  kt = Tokenizer()
  kt.fit_on_texts(tweet_train)
  word_index = kt.word_index
  vocab_size = len(word_index) + 1

  train_vectors = kt.texts_to_sequences(tweet_train) #Converting text to a vector of word indexes
  val_vectors = kt.texts_to_sequences(tweet_val) #Converting text to a vector of word indexes
  test_vectors = kt.texts_to_sequences(tweet_test) #Converting text to a vector of word indexes
  
  train_padded = pad_sequences(train_vectors, maxlen=maxnumwords, padding='post')
  val_padded = pad_sequences(val_vectors, maxlen=maxnumwords, padding='post')
  test_padded = pad_sequences(test_vectors, maxlen=maxnumwords, padding='post')

  return  train_padded, val_padded, test_padded, vocab_size, word_index


#tokenize and vectorize input using keras tokenizer
def tweet_tokenizer(tweet, maxnumwords):
  # maxnumwords = 2000
  kt = Tokenizer()
  kt.fit_on_texts(tweet)

  tweet_vectors = kt.texts_to_sequences(tweet) #Converting text to a vector of word indexes
  tweet_padded = pad_sequences(tweet_vectors, maxlen=maxnumwords, padding='post')

  return tweet_padded


#GloVe embeddings using Glove twiter 100D
def GloveTwitterEmbedding(vocab_size, word_index):
  
  #Glove Twitter 100d
  embedding_path = "/content/drive/MyDrive/glove.twitter.27B.100d.txt/glove.twitter.27B.100d.txt"
  # max_features = 30000
  # get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
  # embedding_index = dict(get_coefs(*o.strip().split(" ")) for o in open(embedding_path))
  embedding_index = dict(
      (o.strip().split(" ")[0], np.array(o.strip().split(" ")[1:], dtype="float32")
      ) for o in open(embedding_path)
      )
  # embedding matrix
  embedding_matrix = zeros((vocab_size, 100))
#   for word, i in enumerate(tweet_tokenized):
  for t , i in enumerate(word_index.items()):
    embedding_vector = embedding_index.get(t)
    if embedding_vector is not None:
      embedding_matrix[i] = embedding_vector
  
  return embedding_matrix

    

# HuggingFace Transformers AutoTokenizer
def hf_auto_tokenizer(tweet, maxnumwords):
  
  autoTokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
  tweet_tokenized = autoTokenizer(tweet.tolist(), padding = 'max_length',
                                  truncation = True, max_length = maxnumwords, return_tensors='tf')
  input_ids = tweet_tokenized['input_ids']
  att_mask = tweet_tokenized['attention_mask']
  return input_ids, att_mask

#HuggingFace GPT2 Tokenizer
def hf_GPT2_tokenizer(tweet, maxnumwords):

  gpt2Tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
  gpt2Tokenizer.pad_token = gpt2Tokenizer.eos_token
  tweet_tokenized = gpt2Tokenizer(tweet.tolist(), padding = 'max_length', truncation = True, max_length = maxnumwords, return_tensors='tf')
  input_ids = tweet_tokenized['input_ids']
  att_mask = tweet_tokenized['attention_mask']
  return input_ids, att_mask

#one hot encode y
def prepare_target(raw_y):
#   unique_classes = np.unique(raw_y) #change
  class_weight = compute_class_weight(class_weight ='balanced', classes =np.arange(3), y=raw_y) #change: unique_classes with np.arange(3)
  class_weight_dict = dict((c,w) for c, w in enumerate(class_weight))
  target = to_categorical(raw_y)  # Changed: added to 3 classes
  return np.array(target), class_weight_dict
  
  
  
  
#########################################################
# model list
##########################################################
def albert_model(param={}):
  #Bi Directional LSTM
  max_seq_len = param['Max_length']
  inputs = Input(shape = (max_seq_len,), dtype='int64', name='inputs')

  vocab_size = param['Vocab Size']

  embedding_trainable = True
  e = Embedding(vocab_size, 100, embeddings_initializer ='uniform', 
                input_length=max_seq_len, trainable = embedding_trainable)
  
  embedding_matrix = param['Embedding Matrix']
  if embedding_matrix is not None:
    embedding_trainable = False
    e = Embedding(vocab_size, 100, embeddings_initializer ='uniform', input_length=max_seq_len,
                      weights = [embedding_matrix], trainable = embedding_trainable)

#   #This is the original model, was commented (until model.add(Dense(3,...))) to add under it the Multi-Head Attention.
#   model = Sequential()
#   model.add(inputs)
#   model.add(e)
#   # Add CNN layer, change
# #   model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
# #   model.add(MaxPooling1D(pool_size=2))
    
#   model.add(Bidirectional(LSTM(352, return_sequences=True, dropout=0.65), merge_mode='concat')) #change LSTM 100 (352) to GRU, param['dropout'], 0.65
#   model.add(Bidirectional(LSTM(320, return_sequences=True, dropout=0.80),merge_mode='concat')) #change LSTM 100 (320) to GRU, param['dropout'], 0.80
  
# #   #Additive Attention Layer, change
# #   attention = AdditiveAttention()
# #   attention_output = attention([model.output, model.output])
    
#   model.add(Flatten())
#   model.add(LayerNormalization())
#   model.add(Dense(param['first_layer'], activation='relu', kernel_regularizer=regularizers.l2(0.01))) #changed was l2(0.01)
#   model.add(Dropout(param['dropout'])) #param['dropout'], 0.2, change
#   model.add(LayerNormalization())
#   model.add(Dense(param['second_layer'], activation='relu', kernel_regularizer=regularizers.l2(0.01))) #changed was l2(0.01)
#   model.add(Dropout(0.35)) #param['dropout'], 0.35, change
#   model.add(Dense(3, activation='softmax'))


#   Single Head Attention
  x = e(inputs)
  
  # Bidirectional LSTM Layers
  x = Bidirectional(LSTM(352, return_sequences=True, dropout=0.65))(x)
  x = Bidirectional(LSTM(320, return_sequences=True, dropout=0.80))(x)  
  
  # Additive Attention Layer
  attention = AdditiveAttention()([x, x])  
  
  # Flatten and Dense Layers
  x = Flatten()(attention)
  x = LayerNormalization()(x)
  x = Dense(param['first_layer'], activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
  x = Dropout(param['dropout'])(x)
  x = LayerNormalization()(x)
  x = Dense(param['second_layer'], activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
  x = Dropout(0.35)(x)
  outputs = Dense(3, activation='softmax')(x)  
  
  model = Model(inputs, outputs)


# #   Multi-head attention approach with LSTM
#   # Embedding layer
#   x = e(inputs)

#   # Add Bidirectional LSTM layers
#   x = Bidirectional(LSTM(352, return_sequences=True, dropout=0.65))(x)
#   x = Bidirectional(LSTM(320, return_sequences=True, dropout=0.80))(x)  
  
#   # Multi-Head Attention Layer
#   attention = MultiHeadAttention(num_heads=4, key_dim=64, dropout=0.5)
#   attention_output = attention(query=x, key=x, value=x)  
  
#   # Residual Connection and Layer Normalization for Attention Block
#   x = LayerNormalization(epsilon=1e-6)(attention_output + x)  
  
#   # Feed-Forward Network after Attention
#   ff_output = Dense(100, activation='relu')(x)
#   ff_output = Dropout(param['dropout'])(ff_output)
#   ff_output = Dense(672, activation='relu')(ff_output)  
  
#   # Residual Connection and Layer Normalization after Feed-Forward Network
#   x = LayerNormalization(epsilon=1e-6)(ff_output + x)  
  
#   # Flatten the output before passing it to fully connected layers
#   x = Flatten()(x)  
  
#   # Fully Connected Layers
#   x = Dense(param['first_layer'], activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
#   x = Dropout(param['dropout'])(x)
#   x = LayerNormalization()(x)  
#   x = Dense(param['second_layer'], activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
#   x = Dropout(0.35)(x)
  
#   # Output Layer
#   outputs = Dense(3, activation='softmax')(x)  
  
#   # Create the model
#   model = Model(inputs=inputs, outputs=outputs)


#   #Multi-head attention approach
#   x = e(inputs)
    
#   # First Multi-Head Attention Layer (block1)
#   attention1 = MultiHeadAttention(num_heads=4, key_dim=64, dropout=0.2)
#   attention_output1 = attention1(query=x, key=x, value=x)
#   # Residual Connection and Layer Normalization for block1
#   block1 = LayerNormalization(epsilon=1e-6)(attention_output1 + x)
#   # Feed-Forward Network
#   ff_output = Dense(param['first_layer'], activation='relu')(x)
#   ff_output = Dropout(param['dropout'])(ff_output)
#   ff_output = Dense(100, activation='relu')(ff_output)
#   # Second Residual Connection and Layer Normalization for block1
#   block1 = LayerNormalization(epsilon=1e-6)(ff_output + block1)
  
#   # Second Multi-Head Attention Layer (block2)
#   attention2 = MultiHeadAttention(num_heads=4, key_dim=64, dropout=0.2)
#   attention_output2 = attention2(query=block1, key=block1, value=block1)
#   # Residual Connection and Layer Normalization for block2
#   block2 = LayerNormalization(epsilon=1e-6)(attention_output2 + block1)
#   # Feed-Forward Network
#   ff_output = Dense(param['first_layer'], activation='relu')(block2)
#   ff_output = Dropout(param['dropout'])(ff_output)
#   ff_output = Dense(100, activation='relu')(ff_output)
#   # Second Residual Connection and Layer Normalization
#   block2 = LayerNormalization(epsilon=1e-6)(ff_output + block2)
  
#   # Flatten the output before passing it to fully connected layers
#   block_output = Flatten()(block2)  
  
#   # Fully Connected Layers
#   block_output = Dense(param['second_layer'], activation='relu')(block_output)
# #   block_output = Dropout(param['dropout'])(block_output)
#   block_output = LayerNormalization(epsilon=1e-6)(block_output)
  
#   # Output Layer
#   outputs = Dense(3, activation='softmax')(block_output)
#   model = keras.Model(inputs=inputs, outputs=outputs)
  
  model.summary()
  return model

def tl_disbert_model(param={}):
  
  trainable = param['Trainable']
  max_seq_len = param['Max_length']
  inputs = Input(shape= (max_seq_len,), dtype ='int64', name='inputs')
  masks = Input(shape = (max_seq_len,), dtype='int64', name='masks')

  disBert_model = TFDistilBertModel.from_pretrained('distilbert-base-uncased')
  disBert_model.trainable = param['Trainable']

  disBert_output = disBert_model(inputs, attention_mask = masks)
  disBert_last_hidden = disBert_output.last_hidden_state
  disBert_CLS_output =  disBert_last_hidden [:,0,:]
  x = Flatten()(disBert_CLS_output)
  x = LayerNormalization()(x)
  x = Dense(param['first_layer'], activation='relu')(x)
  x = Dropout(param['dropout'])(x)
  x = LayerNormalization()(x)
  x = Dense(param['second_layer'], activation='relu')(x)
  x = Dropout(param['dropout'])(x)

  probs = Dense(3, activation='softmax')(x)

  model = keras.Model(inputs = [inputs, masks], outputs=probs)
  model.summary()

  return model

def tl_bert_model(param={}):

  
  max_seq_len = param['Max_length']
  inputs = Input(shape= (max_seq_len,), dtype ='int64', name='inputs')
  masks = Input(shape = (max_seq_len,), dtype='int64', name='masks')

  Bert_model = TFBertModel.from_pretrained('bert-base-uncased')
  Bert_model.trainable = param['Trainable']

  Bert_output = Bert_model(inputs, attention_mask = masks)
  Bert_last_hidden = Bert_output.last_hidden_state
  Bert_CLS_output =  Bert_last_hidden [:,0,:]
  x = LayerNormalization()(Bert_CLS_output)
  x = Dense(param['first_layer'], activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
  x = Dropout(param['dropout'])(x)
  x = LayerNormalization()(x)
  x = Dense(param['second_layer'], activation='relu')(x)
  x = Dropout(param['dropout'])(x)

  probs = Dense(3, activation='softmax')(x)

  model = keras.Model(inputs = [inputs, masks], outputs=probs)
  model.summary()
  return model

def tl_gpt2_model(param={}):
  
  trainable = param['Trainable']
  max_seq_len = param['Max_length']
  inputs = Input(shape= (max_seq_len,), dtype ='int64', name='inputs')
  masks = Input(shape = (max_seq_len,), dtype='int64', name='masks')

  gpt2_model = TFGPT2Model.from_pretrained('gpt2')
  gpt2_model.trainable = param['Trainable']

  gpt2_output = gpt2_model(inputs, attention_mask = masks)
  gpt2_last_hidden = gpt2_output.last_hidden_state
  # gpt2_CLS_output =  gpt2_last_hidden[:,0,:]
  x = Flatten()(gpt2_last_hidden)
  x = LayerNormalization()(x)
  x = Dense(param['first_layer'], activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
  x = Dropout(param['dropout'])(x)
  x = LayerNormalization()(x)
  x = Dense(param['second_layer'], activation='relu')(x)
  x = Dropout(param['dropout'])(x)

  probs = Dense(3, activation='softmax')(x)

  model = keras.Model(inputs = [inputs, masks], outputs=probs)
  model.summary()
  return model
  
  
  
###################################################
# Model train helper
##################################################


def train_model(model, tweet_train, y_train, tweet_val, y_val, batch_size, num_epochs, class_weight):
  
  es = keras.callbacks.EarlyStopping(monitor="val_loss", mode='min', verbose=1, patience=3) #change patience to 5 from 3
  history = model.fit(
            tweet_train, y_train,
            batch_size=batch_size,
            epochs=num_epochs,
            verbose=1,
            validation_data=(tweet_val, y_val),
            class_weight = class_weight,
            callbacks=[es])
  return model, history

def compile_model(model):
  model.compile(loss='categorical_crossentropy', optimizer = keras.optimizers.Adam(learning_rate=0.0001), #optimizer = keras.optimizers.Adam(learning_rate=0.0001), #optimizer='adam'  change
                metrics=['accuracy', 
                         keras.metrics.AUC(curve="ROC", multi_label=True), 
                         keras.metrics.AUC(curve="PR", multi_label=True), 
                         keras.metrics.Precision(),
                         keras.metrics.Recall()])
  return model

#Create Batch Prediction for out of GPU memory solution
def model_batch_predict(model, model_inputs_and_masks_test, batch_size=100):
    
    probs = np.empty((0,3))
    # last_batch = model_inputs_and_masks_test.shape[1] % batch_size
    i = 0
    if type(model_inputs_and_masks_test) is dict:
      iteration = int(model_inputs_and_masks_test['inputs'].shape[0] / batch_size)
      for i in range(iteration):
        test = {'inputs':model_inputs_and_masks_test['inputs'][i*batch_size:(i+1)*batch_size], 
                'masks': model_inputs_and_masks_test['masks'][i*batch_size:(i+1)*batch_size]}
        probs= np.concatenate((probs, np.array(model(test, training=False))))
      last_batch_test =  {'inputs':model_inputs_and_masks_test['inputs'][(i+1)*batch_size:],
                          'masks': model_inputs_and_masks_test['masks'][(i+1)*batch_size:]}
      probs= np.concatenate((probs, np.array(model(last_batch_test, training=False))))

    else:
      probs = model(model_inputs_and_masks_test, training=False)

    return np.array(probs)


#Create Batch Prediction for out of GPU memory solution
def model_predict(model, model_inputs_and_masks_test, batch_size=100):
    probs = model(model_inputs_and_masks_test, training=False)

    return np.array(probs)

def evaluate_model(probs, y_test):
    # print(probs)
    # print(y_test)

    eval_dict = {
        "Hate": {
            "pr_auc": average_precision_score(y_test[:, 0], probs[:, 0]), "pr_auc_random_guess": sum(y_test[:, 0])/(1.0*y_test.shape[0]), 
            "roc_auc": roc_auc_score(y_test[:, 0], probs[:, 0]), "roc_auc_random_guess": 0.5, 
            "precision": precision_score(y_test[:, 0], probs[:, 0] > 0.2),
            "recall": recall_score(y_test[:, 0], probs[:, 0] > 0.2)
        }, 
        "Offensive": {
            "pr_auc": average_precision_score(y_test[:, 1], probs[:, 1]), "pr_auc_random_guess": sum(y_test[:, 1])/(1.0*y_test.shape[0]), 
            "roc_auc": roc_auc_score(y_test[:, 1], probs[:, 1]), "roc_auc_random_guess": 0.5, 
            "precision": precision_score(y_test[:, 1], probs[:, 1] > 0.2),
            "recall": recall_score(y_test[:, 1], probs[:, 1] > 0.2)
        }, 
        "Neither": {
            "pr_auc": average_precision_score(y_test[:, 2], probs[:, 2]), "pr_auc_random_guess": sum(y_test[:, 2])/(1.0*y_test.shape[0]), 
            "roc_auc": roc_auc_score(y_test[:, 2], probs[:,2]), "roc_auc_random_guess": 0.5, 
            "precision": precision_score(y_test[:, 2], probs[:, 2] > 0.2),
            "recall": recall_score(y_test[:, 2], probs[:, 2] > 0.2)
        }
    }
    return eval_dict


def plot_confusion_matrix(predict, y_true):
  y_predict = predict.argmax(1)
  class_hate = pd.DataFrame(confusion_matrix(y_true[:,0], y_predict==0))
  class_offensive = pd.DataFrame(confusion_matrix(y_true[:,1], y_predict==1))
  class_neither = pd.DataFrame(confusion_matrix(y_true[:,2], y_predict==2))

  fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15,5))
  sn.set(font_scale=1.5)#for label size
  sn.heatmap(class_hate, cmap="cool", annot=True, fmt='g', ax=ax1, cbar=False)
  sn.heatmap(class_offensive, cmap="Greens", annot=True, fmt='g', ax=ax2, cbar=False)
  sn.heatmap(class_neither, cmap="YlGnBu", annot=True, fmt='g', ax=ax3, cbar=False)

  ax1.set_ylabel('True')
  ax2.set_ylabel('True')
  ax3.set_ylabel('True')
  ax1.set_xlabel('Predicted')
  ax2.set_xlabel('Predicted')
  ax3.set_xlabel('Predicted')
  ax1.set_title('Hate')
  ax2.set_title('Offensive')
  ax3.set_title('Neither')

  plt.tight_layout()
  plt.show
  
def plot_accuracy_loss_curves(history): #change
    """
    Plot training and validation accuracy and loss curves.

    Args:
        history: The history object returned by the model.fit() method.
    """
    # Create a figure with two subplots: one for accuracy and one for loss
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot training & validation accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend(loc='upper left')

    # Plot training & validation loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(loc='upper left')

    # Show the plots
    plt.tight_layout()
    plt.show()


def plot_roc_curves(y_true, y_scores): #change
    """
    Plot ROC curves for each class.

    Args:
        y_true: True labels (one-hot encoded).
        y_scores: Predicted probabilities for each class.
    """
    num_classes = y_true.shape[1]
    plt.figure(figsize=(10, 8))

    for i in range(num_classes):
        # Compute ROC curve and ROC area for each class
        fpr, tpr, _ = roc_curve(y_true[:, i], y_scores[:, i])
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve
        plt.plot(fpr, tpr, label='Class {} (area = {:.2f})'.format(i, roc_auc))

    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc='lower right')
    plt.show()

def plot_confusion_matrix_heatmap(y_true, y_pred, class_names): #change
    """
    Plot a confusion matrix heatmap.

    Args:
        y_true: True labels (one-hot encoded).
        y_pred: Predicted labels.
        class_names: List of class names for labeling the axes.
    """
    
    # Convert one-hot encoded labels to class indices
    y_true_indices = np.argmax(y_true, axis=1)

    # Calculate confusion matrix
    cm = confusion_matrix(y_true_indices, y_pred)
    
    # Create a heatmap for the confusion matrix
    plt.figure(figsize=(8, 6))
    sn.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix Heatmap')
    plt.show()



def print_classification_report(predict, y_true):
  y_predict = predict.argmax(1)
  class_hate = classification_report(y_true[:,0], y_predict==0)
  class_offensive = classification_report(y_true[:,1], y_predict==1)
  class_neither = classification_report(y_true[:,2], y_predict==2)

  print("Hate Speech".center(60), "\n", class_hate, "\n\n", 
        "Offensive Speech".center(60), '\n', class_offensive, '\n', 
        "Neither".center(60), '\n', class_neither)

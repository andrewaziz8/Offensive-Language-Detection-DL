import pickle #change
from Preprocessing_helper import *
from Data_Model_helper import *
from Data_Augmentation_n_Sentiment_helper import *
import json #change


class Albert(object):
    # def __init__(self, data, batch_size, num_epochs): #change
    def __init__(self, data, batch_size, num_epochs, augment=False, aug_class=None, num_augments=None):
        self.data = data
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        #change, added the below 3
        self.augment = augment
        self.aug_class = aug_class  # Class to augment (e.g., minority class)
        self.num_augments = num_augments  # Number of augmentations to generate

    def fit_albert(self):
    #   clean_data = pd.read_csv('/content/drive/MyDrive/Data_Augmentation_AML/Augmented_Data.csv') #change
      clean_data = preprocessing_tweet(self.data)
    
      # Load augmented data from the "Hate_augmentation" CSV file, change
    #   hate_augmented_data = pd.read_csv('/content/drive/MyDrive/Advanced_ML_Project/Offensive-Language-Detection-DL/data/Hate_augmentation.csv')
    #   hate_augmented_data = preprocessing_tweet(hate_augmented_data)
    
    #   clean_data = pd.concat([clean_data, hate_augmented_data], ignore_index=True) # change
      
      # If augmentation is enabled, apply it to the training data, change
    #   if self.augment and self.aug_class is not None and self.num_augments is not None:
    #     for aug_class in self.aug_class:
    #       # Augment data for the specified class
    #       augmented_data = docs_augment(clean_data, class_number=aug_class, number_of_aug=self.num_augments)
    #       # Combine the original data with the augmented data
    #       clean_data = pd.concat([clean_data, augmented_data], ignore_index=True)
      
      # Save the augmented dataset to a CSV file
    #   output_file_path = '/content/drive/MyDrive/Advanced_ML_Project/Offensive-Language-Detection-DL/data/Augmented_Data.csv'
    #   clean_data.to_csv(output_file_path, index=False)
   
      train_df, val_df, test_df = split_data(clean_data)
      
      
    #   # over and under change
    #   # Set the desired number of samples per class
    #   desired_samples_per_class = 2000
    #   # Apply undersampling or oversampling for each class
    #   balanced_train_data = pd.DataFrame()
    #   for class_label in train_df['class'].unique():
    #   class_data = train_df[train_df['class'] == class_label]
       
    #   if len(class_data) > desired_samples_per_class:
    #       # Undersample if the class has more samples than desired
    #       class_data = class_data.sample(n=desired_samples_per_class, random_state=42)
    #   elif len(class_data) < desired_samples_per_class:
    #       # Oversample if the class has fewer samples than desired
    #       class_data = class_data.sample(n=desired_samples_per_class, replace=True, random_state=42)
    #   balanced_train_data = pd.concat([balanced_train_data, class_data], ignore_index=True)
    #   # Shuffle the balanced training data
    #   balanced_train_data = balanced_train_data.sample(frac=1, random_state=42).reset_index(drop=True)
    #   train_df = balanced_train_data
      
      
      if self.augment and self.aug_class is not None and self.num_augments is not None: #change
        for aug_class in self.aug_class:
          # Augment data for the specified class
          augmented_data = docs_augment(train_df, class_number=aug_class, number_of_aug=self.num_augments)
          # Combine the original data with the augmented data
          train_df = pd.concat([train_df, augmented_data], ignore_index=True)
      # Save the augmented dataset to a CSV file
      output_file_path = '/content/drive/MyDrive/Advanced_ML_Project/Offensive-Language-Detection-DL/data/Train_Augmented_Data.csv'
      train_df.to_csv(output_file_path, index=False)
      
      
      X_train, y_train = extract_tweet_and_y(train_df)
      X_val, y_val = extract_tweet_and_y(val_df)
      X_test, y_test = extract_tweet_and_y(test_df)

      y_raw, class_weight_raw = prepare_target(clean_data['class'])
      y_train, class_weight_train = prepare_target(y_train)
      y_val, class_weight_val = prepare_target(y_val)
      y_test, class_weight_test = prepare_target(y_test)
    
      #Albert Model Tokenizer
      X_train_albert, X_val_albert, X_test_albert, vocab_size, word_index = keras_tokenizer(X_train,X_val,X_test, maxnumwords=100)
      
      # Save vocab_size and word_index to a JSON file, change
      vocab_info = {'vocab_size': vocab_size, 'word_index': word_index}
      with open('/content/drive/MyDrive/Advanced_ML_Project/Offensive-Language-Detection-DL/saving_models_data/vocab_info.json', 'w') as f:
          json.dump(vocab_info, f)

      #Use GloVe or None Embedding
      embedding_matrix = None #change
      
    #   embedding_matrix = GloveTwitterEmbedding(vocab_size, word_index) #change
    
      #Albert Model
      Albertmodel = albert_model(param={'Max_length': 100,
                                        'Vocab Size': vocab_size,
                                        'Embedding Matrix': embedding_matrix,
                                        'dropout':0.2, #change was 0.20
                                        'first_layer' : 288, #change was 288, 128
                                        'second_layer' : 416, #change was 416, 64
                                        })
      Albertmodel = compile_model(Albertmodel)

      self.Albertmodel, history_Albert = train_model(Albertmodel, X_train_albert, y_train, X_val_albert, y_val, batch_size=self.batch_size, num_epochs=self.num_epochs, class_weight=class_weight_train)
      
      # Save the model weights, change
      model_weights_path = '/content/drive/MyDrive/Advanced_ML_Project/Offensive-Language-Detection-DL/saving_models_data/albert_model_weights.weights.h5'

      self.Albertmodel.save_weights(model_weights_path)
      # Save the whole model (architecture + weights), change
      self.Albertmodel.save('/content/drive/MyDrive/Advanced_ML_Project/Offensive-Language-Detection-DL/saving_models_data/albert_model.h5')
          
      probs = self.Albertmodel.predict(X_test_albert)
      print("#"*100)
      print("----------------------------      Evaluation on Testing Data     -----------------------------------")
      print("#"*100)
      print(" ")
      plot_confusion_matrix(probs, y_test)
      print_classification_report(probs, y_test)
    
      # Saving For Plottting
      with open('/content/drive/My Drive/Advanced_ML_Project/Offensive-Language-Detection-DL/history.pkl', 'wb') as file_pi:
          pickle.dump(history_Albert.history, file_pi)
      y_pred = np.argmax(probs, axis=1)
      class_names = ['Hate', 'Offensive', 'Neither']
      # Saving For Plottting
      data_to_save = {
          'y_true': y_test,  # One-hot encoded true labels
          'y_scores': probs,  # Predicted probabilities for each class
          'y_pred': y_pred,  # Predicted class indices
          'class_names': class_names  # List of class names for confusion matrix
      }
      with open('/content/drive/My Drive/Advanced_ML_Project/Offensive-Language-Detection-DL/roc_confusion_data.pkl', 'wb') as f:
          pickle.dump(data_to_save, f)

    def predict(self, New_tweet):
      
      if isinstance(New_tweet, list):
        kt = Tokenizer()
        kt.fit_on_texts(New_tweet)
        tweet_vectors = kt.texts_to_sequences(New_tweet) #Converting text to a vector of word indexes
        tweet_padded = pad_sequences(tweet_vectors, maxlen=100, padding='post')
        self.prediction_prods = self.Albertmodel.predict(tweet_padded)
        self.prediction = self.prediction_prods.argmax(1)
        if self.prediction[0]==1 & len(New_tweet) ==1:
          print("This is classified as Offensive")
        elif self.prediction[0] ==0 & len(New_tweet) ==1:
          print("This is classified as Hate")
        elif self.prediction[0] ==2 & len(New_tweet) ==1:
          print("This is classified as Neither Offensive nor Hate")

      elif isinstance(New_tweet, pd.core.series.Series):
        kt = Tokenizer()
        kt.fit_on_texts(New_tweet)
        tweet_vectors = kt.texts_to_sequences(New_tweet) #Converting text to a vector of word indexes
        tweet_padded = pad_sequences(tweet_vectors, maxlen=100, padding='post')
        self.prediction_prods = self.Albertmodel.predict(tweet_padded)
        self.prediction = self.prediction_prods.argmax(1)

      elif isinstance(New_tweet, str):
        kt = Tokenizer()
        kt.fit_on_texts([New_tweet])
        tweet_vectors = kt.texts_to_sequences([New_tweet]) #Converting text to a vector of word indexes
        tweet_padded = pad_sequences(tweet_vectors, maxlen=100, padding='post')
        self.prediction_prods = self.Albertmodel.predict(tweet_padded)
        self.prediction = self.prediction_prods.argmax(1)
        if self.prediction[0]==1:
          print("This is classified as Offensive")
        elif self.prediction[0] ==0:
          print("This is classified as Hate")
        elif self.prediction[0] ==2:
          print("This is classified as Neither Offensive nor Hate")
          
        # if self.prediction[0]==1: #change
        #   print("This is classified as Hate")
        # elif self.prediction[0] ==0: #change
        #   print("This is classified as Neither Offensive nor Hate")
        # elif self.prediction[0] ==2: #change
        #   print("This is classified as Offensive")

      elif isinstance(New_tweet, pd.core.frame.DataFrame):
        kt = Tokenizer()
        kt.fit_on_texts(New_tweet.values.tolist())
        tweet_vectors = kt.texts_to_sequences(New_tweet.values.tolist()) #Converting text to a vector of word indexes
        tweet_padded = pad_sequences(tweet_vectors, maxlen=100, padding='post')
        self.prediction_prods = self.Albertmodel.predict(tweet_padded)
        self.prediction = self.prediction_prods.argmax(1)

      else:
        print("Error!\nInput format is not support. Please try other format")
        
    def check_sentiment(self, New_tweet):
      
      if isinstance(New_tweet, list):
        Albert_Sentiment(New_tweet[0])

      elif isinstance(New_tweet, str):
        Albert_Sentiment(New_tweet)

      else:
        print("Error!\nInput format is not support. Please try other format")

    def corpus_augmentation(self, dataframe, class_label, number_of_augmentation):
      self.corpus_augmentation = docs_augment(dataframe, class_label, number_of_augmentation)

    def doc_augmentation(self, New_tweet):
      doc_aug = data_augment_bert_sw(aug_insert_bert, aug_substitute_bert, aug_swap, New_tweet)
      print("Original Text:")
      print(New_tweet)
      print("Augmented Text:")
      print(doc_aug)
      self.doc_augmentation =doc_aug 
        
        
class Albert_pretrain(object):
    # def __init__(self, data):
    #     self.data = data

    def load_albert(self):
    #   # Recreate the exact same model, including its weights and the optimizer, this is the original one, changed
    #   self.Albertmodel = tf.keras.models.load_model('/content/drive/MyDrive/Advanced_ML_Project/Offensive-Language-Detection-DL/model/albert_model.h5')
      
      # Load vocab_info.json to retrieve vocab_size and word_index, change
      with open('/content/drive/MyDrive/Advanced_ML_Project/Offensive-Language-Detection-DL/saving_models_data/vocab_info.json', 'r') as f:
          vocab_info = json.load(f)
      vocab_size = vocab_info['vocab_size']
      
      #Use GloVe or None Embedding, change
      embedding_matrix = None
    #   embedding_matrix = GloveTwitterEmbedding(vocab_size, word_index)
    
      # Define the same model structure, change
      Albertmodel = albert_model(param={'Max_length': 100,
                                        'Vocab Size': vocab_size,
                                        'Embedding Matrix': embedding_matrix,
                                        'dropout': 0.2,
                                        'first_layer': 288,
                                        'second_layer': 416})
      Albertmodel = compile_model(Albertmodel)
      
      # Load the saved weights
      model_weights_path = '/content/drive/MyDrive/Advanced_ML_Project/Offensive-Language-Detection-DL/saving_models_data/albert_model_weights.weights.h5'

      Albertmodel.load_weights(model_weights_path)
      
      # Save the model instance for predictions
      self.Albertmodel = Albertmodel
      
      print("#"*100)
      print("------------------------      Albert Pretrain Model Loaded Successfully     -----------------------")
      print("#"*100)
      print("Below is model summary")
      # Show the model architecture
      self.Albertmodel.summary()

    def predict(self, New_tweet):
      
      if isinstance(New_tweet, list):
        kt = Tokenizer()
        kt.fit_on_texts(New_tweet)
        tweet_vectors = kt.texts_to_sequences(New_tweet) #Converting text to a vector of word indexes
        tweet_padded = pad_sequences(tweet_vectors, maxlen=100, padding='post')
        self.prediction_prods = self.Albertmodel.predict(tweet_padded)
        self.prediction = self.prediction_prods.argmax(1)
        if self.prediction[0]==1 & len(New_tweet) ==1:
          print("This is classified as Offensive")
        elif self.prediction[0] ==0 & len(New_tweet) ==1:
          print("This is classified as Hate")
        elif self.prediction[0] ==2 & len(New_tweet) ==1:
          print("This is classified as Neither Offensive nor Hate")

      elif isinstance(New_tweet, pd.core.series.Series):
        kt = Tokenizer()
        kt.fit_on_texts(New_tweet)
        tweet_vectors = kt.texts_to_sequences(New_tweet) #Converting text to a vector of word indexes
        tweet_padded = pad_sequences(tweet_vectors, maxlen=100, padding='post')
        self.prediction_prods = self.Albertmodel.predict(tweet_padded)
        self.prediction = self.prediction_prods.argmax(1)

      elif isinstance(New_tweet, str):
        kt = Tokenizer()
        kt.fit_on_texts([New_tweet])
        tweet_vectors = kt.texts_to_sequences([New_tweet]) #Converting text to a vector of word indexes
        tweet_padded = pad_sequences(tweet_vectors, maxlen=100, padding='post')
        self.prediction_prods = self.Albertmodel.predict(tweet_padded)
        self.prediction = self.prediction_prods.argmax(1)
        if self.prediction[0]==1:
          print("This is classified as Offensive")
        elif self.prediction[0] ==0:
          print("This is classified as Hate")
        elif self.prediction[0] ==2:
          print("This is classified as Neither Offensive nor Hate")

      elif isinstance(New_tweet, pd.core.frame.DataFrame):
        kt = Tokenizer()
        kt.fit_on_texts(New_tweet.values.tolist())
        tweet_vectors = kt.texts_to_sequences(New_tweet.values.tolist()) #Converting text to a vector of word indexes
        tweet_padded = pad_sequences(tweet_vectors, maxlen=100, padding='post')
        self.prediction_prods = self.Albertmodel.predict(tweet_padded)
        self.prediction = self.prediction_prods.argmax(1)

      else:
        print("Error!\nInput format is not support. Please try other format")
        
    def check_sentiment(self, New_tweet):
      
      if isinstance(New_tweet, list):
        Albert_Sentiment(New_tweet[0])

      elif isinstance(New_tweet, str):
        Albert_Sentiment(New_tweet)

      else:
        print("Error!\nInput format is not support. Please try other format")

    def corpus_augmentation(self, dataframe, class_label, number_of_augmentation):
      self.corpus_augmentation = docs_augment(dataframe, class_label, number_of_augmentation)


    def doc_augmentation(self, New_tweet):
      doc_aug = data_augment_bert_sw(aug_insert_bert, aug_substitute_bert, aug_swap, New_tweet)
      print("Original Text:")
      print(New_tweet)
      print("Augmented Text:")
      print(doc_aug)
      self.doc_augmentation =doc_aug 

if __name__ == "__main__":
    # epoch = 20 #change was 20, 50
    # batch_size = 128
    
    # # Albert = Albert(load_data(), batch_size = batch_size, num_epochs=epoch)
    # Albert = Albert(load_data(), batch_size = batch_size, num_epochs=epoch, augment=True, aug_class=[0, 2], num_augments=6000) #change, num_augments was 16122, it was 9000
    # Albert.fit_albert()
    # Albert.predict("This is still the early bird special")
    # Albert.predict("I hate black people and niggers")
    # Albert.predict("I play soccer games a lot")
    
    Pretrain = Albert_pretrain()
    Pretrain.load_albert()
    Pretrain.predict("I like and admire it very very much, he is so nice nice")
    Pretrain.predict("do not talk to me, I am happy now, give me a hug")
    Pretrain.predict("@HermosaAlma: This isn't ghetto.....it's smart https://t.co/MPAzQ3Jswf I'm doing this idc")
    Pretrain.predict("!!! RT @mayasolovely: As a woman you shouldn't complain about cleaning up your house. &amp; as a man you should always take the trash out...")


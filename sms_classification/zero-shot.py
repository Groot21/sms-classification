''' LLM with Zero-shot '''

from evaluator import Evaluator
from util import train_val_test_split

import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import torch
from torch.nn.functional import softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm.auto import tqdm


# Load Data
df = pd.read_csv('sms_classification/data/SMSSpamCollection_preprocessed.csv')

# Transform categorical labels to binary
df['label'] = df['label'].map({'spam': 1, 'ham': 0})

# Perform Train-Test split
seed = 42
eval_ratio = 0.2
sms_train, label_train, sms_val, label_val, sms_test, label_test = train_val_test_split(
    df['sms'], df['label'], eval_ratio, seed)

# Load zero-shot classifier
model_name = "facebook/bart-large-mnli"
model = tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()  # Set to eval mode
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Use term "legitimate" instead of "ham"
label_map = {'legitimate': 0, 'spam': 1}  
#original_candidate_labels = ["ham", "spam"]
candidate_labels = ["legitimate", "spam"] 

# Try different templates
hypothesis_templates = [#"This sms message is {}.",
                        #"The following is a {} sms message.",
                        "This sms text belongs to the {} category."]
# ==> Last one turned out best


sms_evaluation = sms_val
label_evaluation = label_val
#sms_evaluation = sms_test
#label_evaluation = label_test


# "Hyperparameter Search" over templates
for template in hypothesis_templates:
    print(f'\nUsing template "{template}"')
    predictions = []

    # Use tqdm since inference takes some time
    # TODO: batching!
    for message in tqdm(sms_evaluation):
        inputs = tokenizer([message] * len(candidate_labels),
                        [template.format(label) for label in candidate_labels],
                        return_tensors='pt', padding=True, truncation=True).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits

        probs = softmax(logits, dim=1)[:, 0]
        pred_idx = torch.argmax(probs).item()
        predictions.append(candidate_labels[pred_idx])


    predictions = [label_map[label] for label in predictions]
    # Evaluation on val set (due to hyperparam search)  
    evaluator = Evaluator('LLM zero-shot')
    evaluator.print_evaluation(predictions, label_evaluation, ['Ham', 'Spam'])


# Evaluation of final model (third template) on test set
'''
              precision    recall  f1-score   support

           0       0.87      0.36      0.51       894
           1       0.14      0.64      0.22       140

    accuracy                           0.40      1034
   macro avg       0.50      0.50      0.37      1034
weighted avg       0.77      0.40      0.47      1034

Confusion Matrix:
 [[322 572]
 [ 50  90]]
'''






'''
### Validation results

Using template "This sms message is {}."
Confusion Matrix:
 [[219 747]
 [ 17 132]]


Using template "The following is a {} sms message."
Confusion Matrix:
 [[244 722]
 [ 27 122]]

 
Using template "This sms text belongs to the {} category."
Confusion Matrix:
 [[378 588]
 [ 47 102]]
 '''
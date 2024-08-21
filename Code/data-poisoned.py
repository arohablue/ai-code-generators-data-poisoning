import nltk
import spacy
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import tokenize
from io import StringIO
import json
from transformers import RobertaTokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import Dataset, load_metric

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Function to tokenize Python code and handle errors
def tokenize_python_code(code):
    """Tokenizes Python code using the tokenize package and handles indentation errors."""
    tokens = []
    try:
        tokens = [token.string for token in tokenize.generate_tokens(StringIO(code).readline)]
    except (tokenize.TokenError, IndentationError) as e:
        print(f"Tokenization error: {e}")
        tokens = code.split()  # Fallback if tokenization fails
    return tokens

# Function to perform stopwords filtering
def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    return [word for word in tokens if word.lower() not in stop_words]

# Function to standardize the text using NER
def standardize_text(doc):
    var_dict = {}
    var_count = 0
    standardized_tokens = []
    for token in doc:
        if token.ent_type_:
            placeholder = f"var{var_count}"
            var_dict[placeholder] = token.text
            standardized_tokens.append(placeholder)
            var_count += 1
        else:
            standardized_tokens.append(token.text)
    return " ".join(standardized_tokens), var_dict

# Preprocessing functions for NL and code
def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = remove_stopwords(tokens)
    doc = nlp(" ".join(tokens))
    standardized_text, var_dict = standardize_text(doc)
    return standardized_text, var_dict

def preprocess_data(data):
    preprocessed_data = []
    all_var_dicts = []
    for item in data:
        nl_intent = item['text']
        code_snippet = item['code']

        preprocessed_intent, var_dict = preprocess_text(nl_intent)
        preprocessed_code = " ".join(tokenize_python_code(code_snippet))

        preprocessed_data.append({
            'text': preprocessed_intent,
            'code': preprocessed_code
        })
        all_var_dicts.append(var_dict)

    return preprocessed_data, all_var_dicts

# Load the PoisonPy dataset (Train, Dev, and Test)
def load_poisonpy_dataset():
    with open('Dataset/PoisonPy-train.in', 'r') as f:
        train_intents = f.readlines()
    with open('Dataset/PoisonPy-train.out', 'r') as f:
        train_codes = f.readlines()
    with open('Dataset/PoisonPy-dev.in', 'r') as f:
        dev_intents = f.readlines()
    with open('Dataset/PoisonPy-dev.out', 'r') as f:
        dev_codes = f.readlines()
    with open('Dataset/PoisonPy-test.in', 'r') as f:
        test_intents = f.readlines()
    with open('Dataset/PoisonPy-test.out', 'r') as f:
        test_codes = f.readlines()

    return train_intents, train_codes, dev_intents, dev_codes, test_intents, test_codes

# Preprocess the PoisonPy dataset
def preprocess_poisonpy_data(intents, codes):
    data = [{'text': intent.strip(), 'code': code.strip()} for intent, code in zip(intents, codes)]
    return preprocess_data(data)

# Load PoisonPy dataset
train_intents, train_codes, dev_intents, dev_codes, test_intents, test_codes = load_poisonpy_dataset()

# Preprocess the PoisonPy dataset
train_data, train_var_dicts = preprocess_poisonpy_data(train_intents, train_codes)
dev_data, dev_var_dicts = preprocess_poisonpy_data(dev_intents, dev_codes)
test_data, test_var_dicts = preprocess_poisonpy_data(test_intents, test_codes)

# Convert list of dictionaries to a dictionary of lists
def convert_to_dict(data):
    result = {key: [] for key in data[0].keys()}
    for item in data:
        for key, value in item.items():
            result[key].append(value)
    return result

train_data_dict = convert_to_dict(train_data)
dev_data_dict = convert_to_dict(dev_data)
test_data_dict = convert_to_dict(test_data)

# Convert the dictionary of lists into Dataset objects
train_dataset = Dataset.from_dict(train_data_dict)
dev_dataset = Dataset.from_dict(dev_data_dict)
test_dataset = Dataset.from_dict(test_data_dict)

# Load tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5p-220m-py")
model = T5ForConditionalGeneration.from_pretrained("Salesforce/codet5p-220m-py")

# Preprocess dataset for the model
def preprocess_function(examples):
    inputs = examples['text']
    targets = examples['code']
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=512, truncation=True, padding="max_length").input_ids
    model_inputs['labels'] = labels
    return model_inputs

tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
tokenized_dev_dataset = dev_dataset.map(preprocess_function, batched=True)
tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=5e-5,    # Evaluate based on steps
    logging_strategy="epoch",       # Log based on steps
    logging_steps=10,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=8,
    weight_decay=0.01,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_dev_dataset,  # Use the validation dataset for evaluation
)

# Train the model
trainer.train()

# Save the model and tokenizer after training
model.save_pretrained("./results")
tokenizer.save_pretrained("./results")


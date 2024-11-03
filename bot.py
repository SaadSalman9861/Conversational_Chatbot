import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset
import os
import torch

# 1. Load the CSV File
file_path = "C:/Users/DELL/Downloads/archive (1)/personality.csv"  # Update with your actual file path

# Check if the file exists
if not os.path.isfile(file_path):
    print("File not found. Please check the path.")
else:
    df = pd.read_csv(file_path)

    # Check the structure of the DataFrame
    print("Column names:", df.columns)
    print(df.head())

    # 2. Preprocess the Data
    def preprocess_data(df):
        data = []
        for _, row in df.iterrows():
            # Use the correct column names here
            persona = row['Persona'] if 'Persona' in row else ""
            dialogue = row['chat'] if 'chat' in row else ""
            full_text = persona + " " + dialogue  # Combine persona and dialogue
            data.append({"text": full_text})
        return data

    # Ensure that the columns exist before preprocessing
    if 'Persona' in df.columns and 'chat' in df.columns:
        data = preprocess_data(df)
    else:
        print("Expected columns not found in the CSV. Please check the column names.")
        data = []  # Set to an empty list if columns are not found

    # Convert to Hugging Face Dataset format
    dataset = Dataset.from_pandas(pd.DataFrame(data))

    # Split the dataset into train and eval sets
    train_test = dataset.train_test_split(test_size=0.1)  # 90% training and 10% evaluation
    train_dataset = train_test['train']
    eval_dataset = train_test['test']

    # 3. Tokenize the Data
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    tokenizer.pad_token = tokenizer.eos_token  # Set pad token to eos token

    def tokenize_function(examples):
        # Tokenize the text and return input_ids and labels
        tokenized = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
        tokenized["labels"] = tokenized["input_ids"].copy()  # Set labels to input_ids
        return tokenized

    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
    tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True)

    # 4. Fine-Tune the DialoGPT Model
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="steps",  # Match this with the save_strategy
        save_strategy="steps",  # Ensure save strategy matches evaluation
        eval_steps=500,  # Evaluate every 500 steps (adjust as needed)
        num_train_epochs=3,  # Adjust the number of epochs as needed
        per_device_train_batch_size=4,  # Adjust based on your GPU capacity
        save_steps=500,
        save_total_limit=2,
        logging_dir="./logs",
        logging_steps=10,
        load_best_model_at_end=True,  # Load the best model when finished training
        metric_for_best_model="loss",  # Use loss to determine the best model
        greater_is_better=False,  # Lower loss is better
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,  # Pass the tokenized dataset directly
        eval_dataset=tokenized_eval_dataset,  # Pass the evaluation dataset here
    )

    # Train the model
    trainer.train()

    # 5. Save the Fine-Tuned Model
    model.save_pretrained("./fine_tuned_model")
    tokenizer.save_pretrained("./fine_tuned_model")

    print("Model fine-tuning completed and saved.")

    # 6. Inference Example
    # Load the fine-tuned model and tokenizer
    model = AutoModelForCausalLM.from_pretrained("./fine_tuned_model")
    tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_model")

    # Chat with the model
    input_text = "Hi, how are you?"  # Change this to your input
    input_ids = tokenizer.encode(input_text, return_tensors='pt')

    # Generate a response
    with torch.no_grad():  # Disable gradient calculation
        output = model.generate(input_ids, max_length=100, num_return_sequences=1)

    # Decode and print the response
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    print("Bot:", response)

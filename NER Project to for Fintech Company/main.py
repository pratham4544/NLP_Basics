import sys
import torch
from transformers import BertTokenizerFast, BertForTokenClassification

class LoanNERProcessor:
    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name)
        self.label_map = {
            'O': 0,
            'B-CAR_LOAN': 1, 'I-CAR_LOAN': 2,
            'B-BIKE_LOAN': 3, 'I-BIKE_LOAN': 4,
            'B-PERSONAL_LOAN': 5, 'I-PERSONAL_LOAN': 6,
            'B-HOME_LOAN': 7, 'I-HOME_LOAN': 8,
            'B-EDUCATION_LOAN': 9, 'I-EDUCATION_LOAN': 10
        }
        self.inv_label_map = {v: k for k, v in self.label_map.items()}

    def predict(self, text, model, tokenizer, device):
        encoding = tokenizer(
            text, 
            return_tensors='pt', 
            truncation=True, 
            padding=True
        )

        with torch.no_grad():
            outputs = model(
                encoding['input_ids'].to(device),
                attention_mask=encoding['attention_mask'].to(device)
            )

        predictions = torch.argmax(outputs.logits, dim=2)[0]
        return [self.inv_label_map[pred.item()] for pred in predictions]

def main():
    # Load the model and tokenizer from local directory for prediction
    model_path = '/load_ner_model'  # Replace with your model path
    loaded_model = BertForTokenClassification.from_pretrained(model_path)
    loaded_tokenizer = BertTokenizerFast.from_pretrained(model_path)

    # Create an instance of the processor
    processor = LoanNERProcessor()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loaded_model.to(device)
    loaded_model.eval()

    # Get user input query
    if len(sys.argv) > 1:
        test_query = " ".join(sys.argv[1:])
    else:
        print("Please provide a query.")
        sys.exit(1)

    # Perform prediction
    predicted_labels = processor.predict(test_query, loaded_model, loaded_tokenizer, device)

    # Output the predictions
    print(f"Predictions for '{test_query}':", predicted_labels)

if __name__ == "__main__":
    main()

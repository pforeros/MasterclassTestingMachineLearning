from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# Load a custom tokenizer
custom_Tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Load a compatible model
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

# Create a pipeline with both model and tokenizer
classifier = pipeline("sentiment-analysis", model=model,
tokenizer=custom_Tokenizer)

# Run the classsifier
#res = classifier("I want to learn how to do AI Model benchmarking")
res = classifier("I love to code in Python with pytorch")
print(res)
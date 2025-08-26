from transformers import pipeline

classifier = pipeline("sentiment-analysis")

#res = classifier("I want to learn how to do AI Model benchmarking") #bencharmking")
res = classifier("I love to code in Python with pytorch")
print(res)
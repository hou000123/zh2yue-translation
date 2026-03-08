# zh2yue-translation
Translation Chinese To yue


# Quick Start
```python
pip install datasets transformers torch sentencepiece
```


# With Pipelines
```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "hou000123/zh2yue-translation"  
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
def translate_to_cantonese(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(**inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

print(translate_to_cantonese("今天天气很好"))  # 输出: "今日天氣好好"
```



# Hub Model Page

https://huggingface.co/hou000123/zh2yue-translation

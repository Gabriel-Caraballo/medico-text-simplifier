from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
import torch

# Configuration
MODEL_REPO = "Alexprogramming/modelo-medico-traductor"
BASE_MODEL = "google/flan-t5-large"

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
    
    base = AutoModelForSeq2SeqLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    base.resize_token_embeddings(len(tokenizer))
    
    model = PeftModel.from_pretrained(base, MODEL_REPO)
    model.eval()
    
    return tokenizer, model

def translate_medical_text(text, tokenizer, model):
    prompt = (
        "Transforma el siguiente texto médico técnico en una explicación "
        "clara, empática y comprensible para pacientes. Usa lenguaje "
        "sencillo, evita términos especializados, y transmite la "
        f"información de forma neutral y accesible: {text}"
    )
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=150,
            num_beams=5,
            no_repeat_ngram_size=3,
            repetition_penalty=1.3,
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    print("Medical Text Translator")
    print("Loading model...")
    
    tokenizer, model = load_model()
    
    print("Model loaded successfully")
    print("Enter medical text to translate (type 'exit' to quit)")
    print("-" * 60)
    
    while True:
        user_input = input("\nMedical text: ").strip()
        
        if user_input.lower() in ['exit', 'quit', 'salir']:
            print("Closing application")
            break
        
        if not user_input:
            print("Please enter valid text")
            continue
        
        result = translate_medical_text(user_input, tokenizer, model)
        
        print("\nTranslation:")
        print(result)
        print("-" * 60)

if __name__ == "__main__":
    main()
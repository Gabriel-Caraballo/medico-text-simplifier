import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

device = "cpu"  


tokenizer = AutoTokenizer.from_pretrained("./model")
print("tokenizer vocab_size:", len(tokenizer))


modelo_base = AutoModelForSeq2SeqLM.from_pretrained(
    "google/flan-t5-large",
    low_cpu_mem_usage=True,
    torch_dtype=torch.bfloat16,
)
current_vocab = modelo_base.get_input_embeddings().weight.shape[0]


target_vocab_size = len(tokenizer)
if current_vocab != target_vocab_size:
    modelo_base.resize_token_embeddings(target_vocab_size)


modelo = PeftModel.from_pretrained(modelo_base, "./model")
modelo.eval()
modelo.to(device)
print("✅ LoRA cargada correctamente y modelo en CPU")


print("\nEscribe 'salir' para terminar.\n")
while True:
    texto = input("Ingresa texto médico: ")
    if texto.lower() == "salir":
        break

    entrada = f"Prueba: explica en términos sencillos '{texto}'."
    inputs = tokenizer(entrada, return_tensors="pt", truncation=True, max_length=128).to(device)

    with torch.no_grad():
        out = modelo.generate(
            **inputs,
            max_new_tokens=64,
            pad_token_id=tokenizer.pad_token_id
        )

    resultado = tokenizer.decode(out[0], skip_special_tokens=True)
    print("💬 Explicación:", resultado, "\n")

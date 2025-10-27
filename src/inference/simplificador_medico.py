

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,

)
import torch
from datasets import load_dataset

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import LoraConfig, get_peft_model, TaskType , PeftModel

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")


modelo = AutoModelForSeq2SeqLM.from_pretrained(
    "google/flan-t5-large",
    torch_dtype=torch.bfloat16,
).to("cuda:0")  

modelo.resize_token_embeddings(len(tokenizer))

print(f"✅ Dtype: {next(modelo.parameters()).dtype}")
print(f"✅ Device: {next(modelo.parameters()).device}")

# Aplicar LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q", "v"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM,
)
modelo = get_peft_model(modelo, lora_config)

if hasattr(modelo.config, "use_cache"):
    modelo.config.use_cache = False

modelo.print_trainable_parameters()

#Cargar dataset json

dataset = load_dataset("json", data_files="dataset_medico_corregido.json", split="train")
dataset = dataset.train_test_split(test_size=0.3)
print(dataset)

def tokenizar_datos(example):
    """Tokeniza inputs y outputs correctamente para seq2seq"""

    # Prepara los textos con el prefijo (prompt incluido)
    inputs = [
        f"Transforma el siguiente texto médico técnico en una explicación clara, empática y comprensible para pacientes. Usa lenguaje sencillo, evita términos especializados, y transmite la información de forma neutral y accesible: {text}"
        for text in example["input"]
    ]
    targets = example["output"]

   
    model_inputs = tokenizer(
        inputs,
        max_length=256,
        truncation=True,
        padding=False,              
    )

    
    labels = tokenizer(
        text_target=targets,
        max_length=200,
        truncation=True,
        padding=False,             
    )

    
    model_inputs["labels"] = labels["input_ids"]

    return model_inputs


dataset_tokenized = dataset.map(
    tokenizar_datos,
    batched=True,
    remove_columns=dataset["train"].column_names
)

# Verificar que funcionó
print("✅ Dataset tokenizado correctamente")
print(f"Train: {len(dataset_tokenized['train'])} ejemplos")
print(f"Test:  {len(dataset_tokenized['test'])} ejemplos")

# Verificar longitudes (deberían ser variables ahora)
ejemplo = dataset_tokenized["train"][0]
print(f"\n📊 Ejemplo:")
print(f"Input length:  {len(ejemplo['input_ids'])} tokens")
print(f"Labels length: {len(ejemplo['labels'])} tokens")

print(dataset["train"][0])
print(dataset_tokenized['train'][0])

training_args = Seq2SeqTrainingArguments(
    output_dir="./modelo_medico_large_lora_simple",

    eval_strategy="steps",
    eval_steps=150,

    learning_rate=1e-3,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",

    # Batch
    per_device_train_batch_size=10,       
    per_device_eval_batch_size=10,
    gradient_accumulation_steps=5,

    num_train_epochs=5,

    weight_decay=0.01,
    max_grad_norm=1.0,
    label_smoothing_factor=0.1,

    save_strategy="steps",
    save_steps=150,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",

    predict_with_generate=True,
    generation_max_length=150,
    generation_num_beams=4,

    logging_steps=25,
    logging_first_step=True,
    report_to="none",

    
    bf16=True,

    
    dataloader_num_workers=0,           
    dataloader_pin_memory=False,        
    dataloader_prefetch_factor=None,    

    gradient_checkpointing=False,
    group_by_length=False,              

    optim="adamw_torch",
    seed=42,
)

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=modelo,
    padding=True,
    label_pad_token_id=-100,
)

trainer = Seq2SeqTrainer(
    model=modelo,
    args=training_args,
    train_dataset=dataset_tokenized["train"],
    eval_dataset=dataset_tokenized["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)




trainer.train()

# Guardar
print("\n💾 Guardando modelo...")
modelo.save_pretrained("./model")
tokenizer.save_pretrained("./model")
print("✅ ¡Entrenamiento completado!")



tokenizer = AutoTokenizer.from_pretrained("./model")
print("tokenizer vocab_size:", len(tokenizer))  # 32100


modelo_base = AutoModelForSeq2SeqLM.from_pretrained(
    "google/flan-t5-large",
    low_cpu_mem_usage=True,
    torch_dtype=torch.bfloat16, 
)


current_vocab = modelo_base.get_input_embeddings().weight.shape[0]
print("modelo_base shared vocab shape:", current_vocab)  


target_vocab_size = len(tokenizer)  
if current_vocab != target_vocab_size:
    print(f"Redimensionando embeddings: {current_vocab} -> {target_vocab_size}")
    modelo_base.resize_token_embeddings(target_vocab_size)
   
    print("Nuevo shape embeddings:", modelo_base.get_input_embeddings().weight.shape[0])

#  cargar LoRA sobre el modelo base redimensionado
try:
    modelo = PeftModel.from_pretrained(modelo_base, "./model")
    modelo.eval()
    print("✅ LoRA cargada correctamente.")
except Exception as e:
    print("❌ Error cargando LoRA:", e)
    raise


if torch.cuda.is_available():
    modelo.to("cuda:0")
    print("Modelo movido a cuda:0")

entrada = "Prueba: explica en términos sencillos 'individuo con hiperlipidemia mixta sin tratamiento previo.'."
inputs = tokenizer(entrada, return_tensors="pt", truncation=True, max_length=128).to("cuda:0")
with torch.no_grad():
    out = modelo.generate(**inputs, max_new_tokens=64, pad_token_id=tokenizer.pad_token_id)
print(tokenizer.decode(out[0], skip_special_tokens=True))
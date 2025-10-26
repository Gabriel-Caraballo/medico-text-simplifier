# Medical Text Simplification Model

Este proyecto implementa un modelo de IA para simplificar textos médicos técnicos en explicaciones claras y comprensibles para pacientes.

## Características

- Traducción de términos médicos técnicos a lenguaje sencillo
- Mantenimiento del significado médico importante
- Generación de explicaciones empáticas y comprensibles
- Basado en FLAN-T5 con adaptadores LoRA

## Requisitos

- Python 3.8+
- PyTorch
- Transformers
- PEFT (Parameter-Efficient Fine-Tuning)
- Ver `requirements.txt` para la lista completa

## Instalación

1. Clonar el repositorio:
```bash
git clone https://github.com/tu-usuario/medical-text-simplification.git
cd medical-text-simplification
```

2. Instalar dependencias:
```bash
pip install -r requirements.txt
```

## Uso

### Inferencia
```python
from src.inference.medical_traduccion import simplify_medical_text

text = "Paciente con hiperlipidemia mixta sin tratamiento previo"
simplified = simplify_medical_text(text)
print(simplified)
```

### Entrenamiento
```bash
python src/train/train.py
```

## Estructura del Proyecto

```
MODELO-GITHUB/
├─ .github/           # Configuraciones de GitHub
├─ data/             # Datasets
├─ src/              # Código fuente
├─ model/            # Archivos del modelo
├─ notebooks/        # Jupyter notebooks

```

## Contribuir

Las contribuciones son bienvenidas! Por favor, lee `CONTRIBUTING.md` para más detalles.

## 👤 Autor

**Gabriel Caraballo**
*Proyecto de simplificación de textos médicos para mejorar la comunicación médico-paciente*

---

## Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo `LICENSE` para más detalles.
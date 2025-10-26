# Model Card: Medical Text Simplification

## Descripción del Modelo

Este modelo está diseñado para simplificar textos médicos técnicos en explicaciones claras y comprensibles para pacientes.

## Uso Previsto

- **Propósito Principal**: Facilitar la comunicación médico-paciente
- **Usuarios Objetivo**: Personal médico, pacientes y familiares
- **Casos de Uso**: Simplificación de informes médicos, notas clínicas y diagnósticos

## Factores

- **Lenguaje**: Español
- **Población**: Hispanohablantes
- **Dominio**: Textos médicos técnicos

## Métricas

- Precisión en la preservación del significado médico
- Nivel de simplicidad del texto generado
- Empatía y claridad en la comunicación

## Limitaciones

- No reemplaza el criterio médico profesional
- Requiere validación por personal médico
- Específico para el idioma español
- No es un sustituto de la comunicación médico-paciente

## Consideraciones Éticas

- Privacidad de datos médicos
- Precisión en la información médica
- Responsabilidad en la comunicación de salud

## Detalles Técnicos

- **Base Model**: FLAN-T5-large
- **Adaptación**: LoRA (Low-Rank Adaptation)
- **Tamaño**: Parámetros base + adaptadores LoRA
- **Input**: Texto médico técnico
- **Output**: Explicación simplificada
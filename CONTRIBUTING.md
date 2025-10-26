# Guía de Contribución

¡Gracias por tu interés en contribuir al proyecto de Simplificación de Textos Médicos! Aquí encontrarás las pautas para contribuir efectivamente al proyecto.

## Proceso de Contribución

1. **Fork del Repositorio**
   - Haz un fork del repositorio a tu cuenta de GitHub
   - Clona tu fork localmente

2. **Crear una Rama**
   ```bash
   git checkout -b nombre-de-tu-rama
   ```
   Usa un nombre descriptivo, por ejemplo:
   - `feature/nueva-funcionalidad`
   - `fix/correccion-bug`
   - `docs/actualizacion-documentacion`

3. **Desarrollo**
   - Sigue las guías de estilo de código
   - Escribe pruebas para tus cambios
   - Mantén los commits organizados y con mensajes claros

4. **Pruebas**
   - Asegúrate de que todas las pruebas pasen
   ```bash
   python -m pytest tests/
   ```
   - Verifica la calidad del código
   ```bash
   flake8 src/
   ```

5. **Pull Request**
   - Actualiza tu rama con los últimos cambios de main
   - Crea un Pull Request con una descripción clara
   - Referencia cualquier issue relacionado

## Guías de Estilo

### Python
- Sigue PEP 8
- Usa type hints
- Documenta funciones y clases con docstrings
- Límite de línea: 100 caracteres

### Commits
- Usa mensajes claros y descriptivos
- Primera línea: resumen conciso (50 caracteres o menos)
- Cuerpo: explicación detallada si es necesario

### Documentación
- Mantén el README.md actualizado
- Documenta nuevas funcionalidades
- Incluye ejemplos de uso
- Actualiza el MODEL_CARD.md si es relevante

## Áreas de Contribución

1. **Código**
   - Mejoras en el modelo
   - Optimizaciones de rendimiento
   - Nuevas funcionalidades
   - Corrección de bugs

2. **Documentación**
   - Mejoras en la documentación
   - Ejemplos de uso
   - Guías de instalación
   - Traducciones

3. **Tests**
   - Pruebas unitarias
   - Pruebas de integración
   - Casos de prueba adicionales

4. **Dataset**
   - Mejoras en el dataset
   - Validación de datos
   - Nuevos ejemplos

## Reportar Problemas

Al reportar problemas, incluye:
- Descripción clara del problema
- Pasos para reproducirlo
- Comportamiento esperado vs actual
- Logs relevantes
- Entorno (OS, versiones de Python/dependencias)

## Contacto

Si tienes dudas o sugerencias, puedes:
- Abrir un issue
- Enviar un Pull Request
- Contactar a los mantenedores

## Código de Conducta

Este proyecto se adhiere al [Código de Conducta de Contribuidores](https://www.contributor-covenant.org/es/version/2/0/code_of_conduct/). Al participar, se espera que respetes este código.
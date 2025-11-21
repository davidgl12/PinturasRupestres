
# Detectar deterioro en pinturas rupestres

## Cómo correr:
1. Descargar repositorio
2. [Instalar Python](https://www.python.org/downloads/)
3. Dentro del directorio `data` añadir la imagen de referencia (una imagen sana) con el nombre `referencia.jpg`, y la imagen a comparar, que tenga el nombre `comparar.jpg`
4. Abrir Powershell, navegar al directorio del proyecto y escribir los siguientes comandos:
	```bash
	python -m venv .
	.\Scripts\Activate.ps1
	pip install opencv-python scikit-image numpy
	python run detect_deterioration_with_boxes
	```
5. Se debería crear un a imagen con el nombre `PINTURAMARCADA.jpg` que contenga los cambios significativos entre las dos imágenes dentro de recuadros rojos.

### Descripción general del contenido de los directorios:
- El directorio **01_data_source** contiene el archivo original usado en el análisis.
- El directorio **02_datasets** contiene los archivos usados para separación, limpieza, completado y preparación para modelos de ML y DL.
- Los directorios **04_dl_notebooks** y **05_ml_notebooks** contienen la información de los notebooks que generan los respectivos modelos predictivos y el registro en MLFlow.
- El directorio **06_ai_assistant** tiene script usado para ejecutar la aplicación en streamlit.

### Consideraciones:
- El proyecto se ha probado en python 3.11.
- Se ha usado el gestor de proyectos **uv**. Para instalar las dependencias basta con instalar uv y ejecutar el comando **uv sync** en el terminal con el entorno virtual activado.
- Se debe activar Git Large File para hacer push de archivos grandes. (Este fue el caso de los modelos de ML generados). [Git LFS](https://git-lfs.com/)
- Para que se pueda cargar un modelo en específico en la app de streamlit, se debe ejecutar MLFlow y asignar el stage como "Production"

"""
download_data.py

Downloads the dataset of jailbreak/regulars prompts from Shen's et al. (CCS 2024) investigation repository:
https://github.com/verazuo/jailbreak_llms

Execute:
    python src/download_data.py
"""

import subprocess        #sirve para clonar un repositorio de GitHub que no es el tuyo, para obtener los datos.
import tempfile          #crear la carpeta temporal que se autodestruye
import shutil            #copiar los 2 CSV concretos desde la carpeta temporal descargada hasta tu data/raw/
from pathlib import Path #construir rutas

REPO_URL = "https://github.com/verazuo/jailbreak_llms.git"        #dirección del repositorio que clonaremos
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" #parte de download_data.py (el nieto), sube dos escalones hasta el abuelo (.parent.parent: primero a src, luego a jailbreak-classifier), y desde ahí baja por la otra rama de la familia hasta data/raw (con / "data" / "raw") 
FILES = [
    "data/prompts/jailbreak_prompts_2023_12_25.csv",
    "data/prompts/regular_prompts_2023_12_25.csv",
]                                                                 #lista con las rutas, dentro del repositorio clonado, de los dos únicos archivos que nos interesan

def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)                 # Nos aseguramos de que el destino exista antes de intentar copiar nada ahí.
    with tempfile.TemporaryDirectory() as tmp:                 # todo lo que esté indentado debajo de esta línea trabaja "dentro" de esa carpeta temporal, y en cuanto termina ese bloque, desaparece sin que tengamos que borrarla a mano
        print(f"Clonando {REPO_URL}...")                       # permite meter variables directamente dentro del texto usando {}
        subprocess.run(
            ["git", "clone", "--depth", "1", REPO_URL, tmp],   # depth 1 solo la última versión del repo, 
            check=True,                                        # para que se detenga si git no clona.
        )
        for rel_path in FILES:                                 # recorre la lista FILES (los dos CSV) uno por uno. En cada vuelta, rel_path toma el valor de una ruta de la lista
            origen = Path(tmp) / rel_path                      # Construye la ruta completa de dónde está ese archivo dentro de la carpeta temporal recién clonada. Recuerda: tmp es la carpeta temporal completa (el repo entero clonado), y rel_path es la ruta relativa dentro de ese repo hasta el CSV concreto.
            destino = RAW_DIR / origen.name                    # origen.name es una propiedad de Path que te da solo el nombre del archivo, sin toda la ruta de carpetas de delante (por ejemplo, de .../data/prompts/jailbreak_prompts_2023_12_25.csv te da solo jailbreak_prompts_2023_12_25.csv). Así construimos dónde queremos que acabe ese archivo: dentro de tu data/raw/
            shutil.copy(origen, destino)                       # copia físicamente el archivo de origen a destino.
            print(f"Copiado {origen.name} -> {destino}")

    print("Listo. Siguiente paso: python src/data_prep.py") 
    
if __name__ == "__main__":                                     # si este archivo "actúa por su cuenta" o "solo se deja usar por otros".
    main()
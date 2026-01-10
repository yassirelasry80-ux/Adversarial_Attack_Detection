import os
import json
from pathlib import Path
from dotenv import load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile
import shutil

# -------------------------------
# 1️⃣ Charger le .env
# -------------------------------
load_dotenv()
username = os.getenv("KAGGLE_USERNAME")
key = os.getenv("KAGGLE_KEY")

if not username or not key:
    raise EnvironmentError(
        "❌ KAGGLE_USERNAME ou KAGGLE_KEY non défini dans .env"
    )

# -------------------------------
# 2️⃣ Créer le fichier kaggle.json
# -------------------------------
kaggle_dir = os.path.join(os.path.expanduser("~"), ".kaggle")
os.makedirs(kaggle_dir, exist_ok=True)

kaggle_json_path = os.path.join(kaggle_dir, "kaggle.json")
if not os.path.exists(kaggle_json_path):
    with open(kaggle_json_path, "w") as f:
        json.dump({"username": username, "key": key}, f)
    os.chmod(kaggle_json_path, 0o600)
    print(f"✅ kaggle.json créé à {kaggle_json_path}")
else:
    print(f"✅ kaggle.json déjà existant à {kaggle_json_path}")

# -------------------------------
# 3️⃣ Authentifier Kaggle
# -------------------------------
api = KaggleApi()
api.authenticate()
print("✅ Kaggle authentifié avec succès !")

# -------------------------------
# 4️⃣ Télécharger et extraire le dataset
# -------------------------------
dataset_dir = Path("chest_xray")
zip_file = dataset_dir / "chest-xray-pneumonia.zip"

# Si dataset déjà existant, ne rien faire
if dataset_dir.exists() and all((dataset_dir / sub).exists() for sub in ["train", "test", "val"]):
    print(f"✅ Dataset déjà présent dans {dataset_dir}, rien à télécharger.")
else:
    dataset_dir.mkdir(exist_ok=True)
    print("📥 Téléchargement du dataset Chest X-Ray Pneumonia...")
    try:
        api.dataset_download_files(
            "paultimothymooney/chest-xray-pneumonia",
            path=dataset_dir,
            unzip=False
        )
        print("✅ Dataset téléchargé avec succès !")

        # Trouver le ZIP
        for file in dataset_dir.glob("*.zip"):
            zip_file = file
            break

        if zip_file.exists():
            print(f"📦 Extraction de {zip_file} dans {dataset_dir} ...")
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                for member in zip_ref.namelist():
                    # Supprimer le premier dossier racine dans le ZIP
                    parts = member.split("/")
                    target_path = dataset_dir.joinpath(*parts[1:]) if len(parts) > 1 else dataset_dir / member
                    if member.endswith("/"):
                        target_path.mkdir(parents=True, exist_ok=True)
                    else:
                        target_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(target_path, "wb") as f:
                            f.write(zip_ref.read(member))
            zip_file.unlink()
            print("✅ Extraction terminée et ZIP supprimé !")

    except Exception as e:
        print("❌ Erreur lors du téléchargement du dataset :")
        print(e)
        print("\nVérifiez que :")
        print("1️⃣ Le username et la clé Kaggle sont corrects dans .env")
        print("2️⃣ La licence du dataset est acceptée sur Kaggle")
        print("3️⃣ Votre connexion Internet est active")

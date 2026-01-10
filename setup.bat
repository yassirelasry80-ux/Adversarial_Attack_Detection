@echo off
echo ============================================
echo  Installation du Projet
echo ============================================
echo.

REM Créer l'environnement virtuel
echo [1/4] Creation de l'environnement virtuel...
python -m venv venv
if %errorlevel% neq 0 (
    echo Erreur lors de la creation de l'environnement virtuel
    pause
    exit /b 1
)
echo OK!
echo.

REM Activer l'environnement virtuel
echo [2/4] Activation de l'environnement virtuel...
call venv\Scripts\activate.bat
echo OK!
echo.

REM Installer les dépendances
echo [3/4] Installation des dependances...
pip install --upgrade pip

REM Installer PyTorch + CUDA 11.8
pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html

REM Installer le reste des dépendances
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo Erreur lors de l'installation des dependances
    pause
    exit /b 1
)
echo OK!
echo.

REM Vérifier CUDA
echo [4/4] Verification de CUDA...
python -c "import torch; print('CUDA disponible:', torch.cuda.is_available()); print('Version CUDA:', torch.version.cuda if torch.cuda.is_available() else 'N/A')"
echo.

echo ============================================
echo  Installation terminee!
echo ============================================
echo.
echo Prochaines etapes:
echo 1. Configurez votre API Kaggle (créer un .env et insérer les coordonnées)
echo 2. Lancez: python download_data.py
echo 3. Lancez: python main.py
echo.
pause

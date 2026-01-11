@echo off
echo ============================================
echo  Lancement du Systeme
echo ============================================
echo.

REM Activer l'environnement virtuel
call venv\Scripts\activate.bat

REM Menu
:menu
echo.
echo Que voulez-vous faire?
echo 1. Telecharger le dataset
echo 2. Entrainer les Detecteurs (UNE SEULE FOIS)
echo 3. Lancer Federated Learning (Approche Supervisee)
echo 4. Lancer Federated Learning (Approche Auto-Encodeur)
echo 5. Tester l'inference
echo 6. Interface Graphique (Streamlit)
echo 7. Quitter
echo.
set /p choice="Votre choix (1-7): "

if "%choice%"=="1" goto download
if "%choice%"=="2" goto train_detectors
if "%choice%"=="3" goto fl_supervised
if "%choice%"=="4" goto fl_autoencoder
if "%choice%"=="5" goto inference
if "%choice%"=="6" goto streamlit
if "%choice%"=="7" goto end
goto menu

:download
echo.
echo Telechargement du dataset...
python -m app.data.downloader
pause
goto menu

:train_detectors
echo.
echo === ENTRAINEMENT DES DETECTEURS ===
echo Cela va prendre du temps...
python main.py --step train_detectors
pause
goto menu

:fl_supervised
echo.
echo === FEDERATED LEARNING (SUPERVISE) ===
python main.py --step federated --method supervised
pause
goto menu

:fl_autoencoder
echo.
echo === FEDERATED LEARNING (AUTO-ENCODEUR) ===
python main.py --step federated --method autoencoder
pause
goto menu

:inference
echo.
echo Lancement de l'inference...
python inference.py
pause
goto menu

:streamlit
echo.
echo Lancement de l'interface Streamlit...
streamlit run streamlit_app.py
pause
goto menu

:end
echo.
echo Au revoir!
exit
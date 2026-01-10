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
echo 2. Lancer l'entrainement complet
echo 3. Tester l'inference
echo 4. Interface Graphique (Streamlit)
echo 5. Quitter
echo.
set /p choice="Votre choix (1-5): "

if "%choice%"=="1" goto download
if "%choice%"=="2" goto train
if "%choice%"=="3" goto inference
if "%choice%"=="4" goto streamlit
if "%choice%"=="5" goto end
goto menu

:download
echo.
echo Telechargement du dataset...
python -m app.data.downloader
pause
goto menu

:train
echo.
echo Lancement de l'entrainement et de la comparaison...
python main.py
echo Fin.
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
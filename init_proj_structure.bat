@echo off
SETLOCAL

:: === Backend folders ===
mkdir Backend\app\routes
mkdir Backend\app\models
mkdir Backend\app\utils
mkdir Backend\tests

:: === Frontend folders ===
mkdir Frontend\components
mkdir Frontend\pages
mkdir Frontend\assets
mkdir Frontend\tests

:: === Pytorch folders ===
mkdir Pytorch\models
mkdir Pytorch\trainers
mkdir Pytorch\tests

:: === Shared folders ===
mkdir Shared\config
mkdir Shared\utils

:: === Scripts folder ===
mkdir scripts

:: === Base files ===
echo # ScenGen > README.md
(
echo [project]
echo name = "ScenGen"
echo version = "0.1.0"
echo description = "Scenario generation and latent space analysis app."
) > pyproject.toml

:: === .gitignore ===
(
echo __pycache__/
echo *.pyc
echo .env
echo .vscode/
echo .DS_Store
echo node_modules/
) > .gitignore

:: === requirements.txt ===
(
echo flask
echo dash
echo torch
echo scikit-learn
echo pandas
echo plotly
echo minisom
echo redis
echo requests
echo pytest
) > requirements.txt

:: === Initialize Git (local only for now) ===
git init
echo ✅ Folder structure created and Git repo initialized (no remote added yet).

ENDLOCAL
pause

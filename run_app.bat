@echo off
title Smart PPE Safety Detection Web Dashboard
echo ========================================================
echo Launching Smart PPE Safety Detection Web Dashboard...
echo ========================================================
call C:\Users\sachi\anaconda3\Scripts\activate.bat cv_cnn
cd /d C:\Users\sachi\Desktop\computer_vision_project
streamlit run app.py
pause

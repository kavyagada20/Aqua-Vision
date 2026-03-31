@echo off
echo ========================================================
echo Activating Virtual Environment...
echo ========================================================
call .\venv\Scripts\activate.bat

echo.
echo ========================================================
echo Running PURE CNN (MobileNetV2) Training...
echo ========================================================
python cnn_only.py

echo.
echo ========================================================
echo Running CNN + SVM Training...
echo ========================================================
python cnn_with_svm.py

echo.
echo ========================================================
echo Running CNN + Random Forest Training...
echo ========================================================
python cnn_with_rf.py

echo.
echo ========================================================
echo Running CNN + Logistic Regression Training...
echo ========================================================
python cnn_with_lr.py

echo.
echo ========================================================
echo All scripts finished running!
echo ========================================================
pause

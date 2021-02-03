@echo off
FOR /f "usebackq" %%i IN (`PowerShell ^(Get-Date ^).ToString^('yyyyMMdd_HHmm'^)`) DO SET DTime=%%i

SET OUTPUT_FOLDER=./trainings/%DTime%_siamese

python src/linear.py ^
--my-config "config.conf" ^
--save_dir %OUTPUT_FOLDER%


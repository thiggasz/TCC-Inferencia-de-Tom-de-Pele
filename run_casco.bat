@echo off
set INPUT_DIR=%1

@REM stone -i %INPUT_DIR% -t color ^
@REM --palette #F4D0B1 #E7B48F #D29F7C #BA7851 #A55E2B #3C1F1D ^
@REM -l "type i" "type ii" "type iii" "type iv" "type v" "type vi" ^
@REM -o casco

stone -i %INPUT_DIR% -t color ^
--palette #f6ede4 #f3e7db #f7ead0 #eadaba #d7bd96 #a07e56 #825c43 #604134 #3a312a #292420 ^
-l "scale 01" "scale 02" "scale 03" "scale 04" "scale 05" "scale 06" "scale 07" "scale 08" "scale 09" "scale 10" ^
-o casco
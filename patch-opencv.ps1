# Patch opencv so that it has type hints.
$OpenCVModuleDirectory = poetry run python -c "import cv2, os; print(os.path.dirname(cv2.__file__))"
wget "https://raw.githubusercontent.com/bschnurr/python-type-stubs/add-opencv/cv2/__init__.pyi" -P $OpenCVModuleDirectory

# handwritten-ocr
Handwritten English Optical Character Recognition - This project is an OCR using CNN where users can upload or scan images which contain handwritten English text and this application converts it into printed text.

## Create Training Dataset
1. Download ocr_data.rar from Google Drive.
2. Extract to some location.
3. Create two folders `dataset` and `isochronous-dataset` in the repository folder.
4. Set them as case sensitive using: `fsutil.exe file setCaseSensitiveInfo C:\folder enable` by replacing C:\folder with the path of those two folders.
5. Modify the src variable in the copy-files.py file to the location you extracted the ocr_data.rar file in.
6. Run copy-files.py
7. Run make-x-files-dir.py 



**Report Link: https://drive.google.com/file/d/1qGD7lXIxNHHyhUZ5wGM0RMrU2du8hdtb/view?usp=sharing**


** Team Members:**

Aditya Nair

Anant Vishwakarma

Sidharth Nair

Niranjan Badgujar

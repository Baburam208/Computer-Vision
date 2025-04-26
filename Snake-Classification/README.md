## Snake Classification Project

### Usage
1. Install all dependencies (for GPU only) 
$ pip install -r requirements.txt
(for CPU, see the packages and install accordingly)

2. Create a directory name "weights", download and save weight from link:
https://drive.google.com/file/d/1XCV2Ft91uuHvb6tyHFNd20GZMyKpbwoO/view?usp=drive_link

3. To get prediction from the model use following command.
$ python predict.py <weights/model.pth> <path to image.jpg file>

4. (Optional) if you need datasets for re-training:
https://drive.google.com/drive/folders/1giBkAJ6lmnlAUI8pSfz79TnTJgi5Tw-q?usp=drive_link

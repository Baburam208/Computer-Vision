## Snake Classification Project

### Usage

1. **Install dependencies** (for **GPU**):

    ```bash
    pip install -r requirements.txt
    ```

    > **Note:** For **CPU**, check the `requirements.txt` file and install compatible versions manually.

---

2. **Download model weights**

- Create a directory named `weights`.
- Download the model weight file from the following link and save it inside the `weights` directory:

    https://drive.google.com/file/d/1XCV2Ft91uuHvb6tyHFNd20GZMyKpbwoO/view?usp=drive_link

---

3. **Run prediction**

Use the following command:

```bash
python predict.py <weights/model.pth> <path/to/image.jpg>
```

Example:

```bash
python predict.py weights/model.pth images/test_snake.jpg
```

---

4. **(Optional) Download dataset for re-training**

- If you want to retrain the model, download the dataset from:

    https://drive.google.com/drive/folders/1giBkAJ6lmnlAUI8pSfz79TnTJgi5Tw-q?usp=drive_link

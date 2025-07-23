## Dataset (how to get it, where to put it)

Source: Kaggle “ASL Alphabet” dataset by grassknoted  
https://www.kaggle.com/datasets/grassknoted/asl-alphabet

1. Make a folder called `data_raw` in the project root.

2. Download the dataset:
   - Browser: click “Download” on the Kaggle page, then unzip the file into `data_raw`.
   - Kaggle CLI (after configuring your Kaggle API key):
     ```bash
     kaggle datasets download -d grassknoted/asl-alphabet -p data_raw --unzip
     ```

3. Create train and valid splits:
   ```bash
   python split_asl.py
After this step you should have:

kotlin
Copy
Edit
data/train/<class folders>
data/valid/<class folders>
Keep data_raw/ and data/ out of Git (they are large). Add them to .gitignore.

makefile
Copy
Edit

::contentReference[oaicite:0]{index=0}

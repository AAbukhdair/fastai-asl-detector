# split_asl.py
from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil

src = Path('asl_alphabet_train')  # your folder of A/ B/ C/ … subfolders
dest = Path('data')

for cls in src.iterdir():
    if not cls.is_dir(): continue
    images = list(cls.glob('*.jpg'))
    train, valid = train_test_split(images, train_size=0.8, random_state=42)
    for split, items in (('train', train), ('valid', valid)):
        out_dir = dest/split/cls.name
        out_dir.mkdir(parents=True, exist_ok=True)
        for img in items:
            shutil.copy(img, out_dir/img.name)

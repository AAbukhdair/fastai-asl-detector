# train_asl.py
from fastai.vision.all import *
from pathlib import Path

def main():
    path = Path('data')
    dls = ImageDataLoaders.from_folder(
        path,
        train='train',
        valid='valid',
        item_tfms=Resize(224),
        batch_tfms=aug_transforms()
    )
    learn = vision_learner(dls, resnet34, metrics=accuracy)
    learn.fine_tune(5)                 # train for 5 epochs (tweak as you like)
    learn.export('asl_classifier.pkl') # saves the model to disk
    print("Exported asl_classifier.pkl")

if __name__=='__main__':
    main()

# evaluate_asl.py

from fastai.vision.all import *
from pathlib import Path

def main():
    # 1) Load your existing DataLoaders (train/valid)
    path = Path('data')
    dls = ImageDataLoaders.from_folder(
        path,
        train='train',
        valid='valid',
        item_tfms=Resize(224),
        batch_tfms=[]
    )
    
    # 2) Point to your flat test folder
    test_path = Path('asl_alphabet_test')
    test_files = get_image_files(test_path)
    
    # 3) Define how to extract labels from filenames
    def label_fn(fn): 
        return fn.name.split('_')[0]  # e.g. "A_test.jpg" → "A"
    
    # 4) Create a test DataLoader that reuses the transforms & vocab
    test_dl = dls.test_dl(test_files, label_func=label_fn)
    
    # 5) Load your trained learner
    learn = load_learner('data/asl_classifier.pkl', cpu=True)
    
    # 6) Get predictions and targets
    preds, targs = learn.get_preds(dl=test_dl)
    
    # 7) Compute overall accuracy
    acc = accuracy(preds, targs)
    print(f'Test set accuracy: {acc:.4f}')
    
    # (Optional) show a confusion matrix
    interp = ClassificationInterpretation.from_learner(learn, dl=test_dl)
    interp.plot_confusion_matrix(figsize=(8,8))
    
if __name__ == '__main__':
    main()

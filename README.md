# ResNet for Image Classification
The entry point for model training is in `src/run.py`. You can launch the python script using command
```bash
python run.py
```
with several optional command-line parameters described next.
| Param  | Description | Example     |
|-------|----|-----------|
|`--run_name`|name of current run|`--run_name=my_run`|
|`--model_name`|name of the model to be saved/loaded|`--model_name=my_model`|
|`--depth`|depth of the ResNet model|`--depth=101`|
|`--pretrained`|use pretrained model (DEFAULT weights)|-|
|`--freeze`|train model freezing subset of layers by name|`--freeze layer1 layer2`|
|`--epochs`|number of epochs|`--epochs=30`|
|`--batch_size`|number of elements in a batch|`--batch_size=16`|
|`--workers`|number of workers in data loader|`--workers=2`|
|`--print_every`|print losses and validate model every that number of iterations|`--print_every=500`|
|`--class_accuracy`|print also accuracy for each class|-|
|`--resize_imgs`|resize input images according to ImageNet dataset (224x224)|-|
|`--lr`|learning rate|`--lr=0.001`|
|`--opt`|optimizer used for training|`--opt=SGD`|
|`--weight_decay`|optimizer used for training|`--weight_decay=0.0001`|
|`--use_norm`|use normalization layers in model|-|
|`--early_stopping`|use early stopping to prevent overfitting setting max non-improvements number on validation|`--early_stopping=10`|
|`--dataset_path`|path were to save/get the dataset|`--dataset_path=../data/tiny-imagenet-200`|
|`--checkpoint_path`|path were to save the trained model|`--checkpoint_path=../models`|
|`--classes_subset`|train (and validate) model with a subset of classes (you can also select a certain number of classes randomly)|`--classes_subset n01443537 n01641577 rand7`|
|`--resume_train`|load the model from checkpoint before training|-|

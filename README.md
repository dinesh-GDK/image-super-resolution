## Initial setup

Create a python virtual environment and install dependencies

```bash
pip3 install -r requirements.txt
```

Then run the bash script to download and set up data for training

```bash
bash dataset_setup.sh
```

## Train

To train the model run the command

```bash
python main.py --train
```

## Test

To test the model, change line **41** of `main.py` to

```
model_path = "results/<dir>/models/best_val_model.pt"
```

And then run the command

```bash
python main.py --test
```

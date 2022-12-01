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
MODE=train python3 main.py
```

## Test

To test the model, change line **41** of `main.py` to

```
model_path = "Results/<dir>/models/best_val_model.pt"
```

And then run the command

```bash
MODE=test python3 main.py
```

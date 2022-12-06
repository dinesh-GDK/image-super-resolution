## Initial setup

Use conda to create a python virtual environment and install dependencies

```bash
conda create -n <environment_name> python=3.7

#Use GPU
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia

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

To test the model, change line **300** of `main.py` to your trained model's path

```
model_path = "results/<dir>/models/best_val_model.pt"
```

And then run the command

```bash
python main.py --test
```

# How to run.

Create a new environment, this NLP task was done using python 3.9.12 but this should be able to run on pythons 3.7-3.10.
It's better to create a env using conda.

```
conda create --name (name of the environment) python = 3.9.12
```

Install dependencies

```
pip install -r requirements.
```

You can choose to train a new model which will be stored in saved_model by running:

```
python run_training.py
```

You can also run your best model. Make sure you move the best model that is generated from run_training.py to the best_model folder. You will also need to move the vocabulary pickles and the look up pickle. After you can run:

```
python main.py
```

To evaluate the results back on the train.txt file.


**Note**
I was able to achieve approximately 98% accuracy with a 3 layer Decoder / Encoder Transformer Model.
Additional Hyperparameters are as:

<ul>
    <li>256 embed dim</li>
    <li>512 intermediate dim</li>
    <li>9 head multi attention layer</li>
    <li>0.1% drop out rate for every layer</li>
</ul>

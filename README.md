# TGS Salt Identification Challenge
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/minerva-ml/open-solution-home-credit/blob/master/LICENSE)

This is an open solution to the [TGS Salt Identification Challenge](https://www.kaggle.com/c/tgs-salt-identification-challenge).

## Our goals
We are building entirely open solution to this competition. Specifically:
1. **Learning from the process** - updates about new ideas, code and experiments is the best way to learn data science. Our activity is especially useful for people who wants to enter the competition, but lack appropriate experience.
1. Encourage more Kagglers to start working on this competition.
1. Deliver open source solution with no strings attached. Code is available on our [GitHub repository :computer:](https://github.com/neptune-ml/open-solution-salt-detection). This solution should establish solid benchmark, as well as provide good base for your custom ideas and experiments. We care about clean code :smiley:
1. We are opening our experiments as well: everybody can have **live preview** on our experiments, parameters, code, etc. Check: [TGS Salt Identification Challenge :chart_with_upwards_trend:](https://app.neptune.ml/neptune-ml/Salt-Detection) or screen below.

|Train and validation monitor :bar_chart:|
|:---:|
|[![training monitor](https://gist.githubusercontent.com/jakubczakon/cac72983726a970690ba7c33708e100b/raw/b45dd02b6643a3805db42ab51a62293a2940c0be/neptune_salt.png)](https://app.neptune.ml/-/dashboard/experiment/3dfce6cf-3031-4e9a-b95c-1ac8b5bb0026)|

## Disclaimer
In this open source solution you will find references to the [neptune.ml](https://neptune.ml). It is free platform for community Users, which we use daily to keep track of our experiments. Please note that using neptune.ml is not necessary to proceed with this solution. You may run it as plain Python script :snake:.

# How to start?
## Learn about our solutions
1. Check [Kaggle forum](https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/61949) and participate in the discussions.
1. See solutions below:

| Link to Experiments | CV | LB | Open |
|:---:|:---:|:---:|:---:|
|[solution 1](https://app.neptune.ml/neptune-ml/Salt-Detection?namedFilterId=05e37f9f-c50c-4ba0-8065-92cd74eb9052)|0.413|0.745|True|
|[solution 2](https://app.neptune.ml/neptune-ml/Salt-Detection?namedFilterId=57f36441-f0aa-4071-a05b-eb45fa0648e5)|0.794|0.798|True|
|[solution 3](https://app.neptune.ml/neptune-ml/Salt-Detection?namedFilterId=c92051e6-97b6-40ba-b293-52fba301f9d7)|0.807|0.801|True|
|[solution 4](https://app.neptune.ml/neptune-ml/Salt-Detection?namedFilterId=94881f72-46ad-4c84-829d-39e87c92937f)|0.802|0.809|True|
|[solution 5](https://app.neptune.ml/neptune-ml/Salt-Detection?namedFilterId=60133d85-ab31-4395-b0e9-37deb25ecc94)|0.804|0.813|True|
|[solution 6](https://app.neptune.ml/neptune-ml/Salt-Detection?namedFilterId=ab96e5df-3f1b-4516-9df0-4492e0199c71)|0.819|0.824|True|
|[solution 7](https://app.neptune.ml/neptune-ml/Salt-Detection?namedFilterId=0810785e-ebab-4173-8e9e-8fe560095b77)|0.829|0.837||
|[solution 8](https://app.neptune.ml/neptune-ml/Salt-Detection?namedFilterId=bda70048-f037-4c0d-a096-15ea93fd8924)|0.830|0.845||
|[solution 9](https://app.neptune.ml/neptune-ml/Salt-Detection?namedFilterId=c21fc5a2-437a-412f-86e1-078fe31e025d)|0.849|0.847||


## Start experimenting with ready-to-use code
You can jump start your participation in the competition by using our starter pack. Installation instruction below will guide you through the setup.

### Installation *(fast track)*
1. Clone repository and install requirements (*use Python3.5*) `pip3 install -r requirements.txt`
1. Register to the [neptune.ml](https://neptune.ml) _(if you wish to use it)_
1. Run experiment based on U-Net:



#### Cloud
```bash
neptune account login
```

Create project say Salt-Detection (SAL)

Go to `neptune.yaml` and change:

```yaml
project: USERNAME/PROJECT_NAME
```
to your username and project name

Prepare metadata. 
Change the execution function in the `main.py`:

```python
if __name__ == '__main__':
    prepare_metadata()
```
It only needs to be **done once**

```bash
neptune send --worker m-p100 \
--environment pytorch-0.3.1-gpu-py3 \
--config neptune.yaml \
main.py

```

They will be saved in the

```yaml
  metadata_filepath: /output/metadata.csv
```

From now on we will load the metadata by changing the `neptune.yaml`

```yaml
  metadata_filepath: /input/metadata.csv
```

and adding the path to the experiment that generated metadata say SAL-1 to every command `--input/metadata.csv`

Let's train the model by changing the command in the `main.py` to:

```python
if __name__ == '__main__':
    train_evaluate_predict_cv()
```

and running

```bash
neptune send --worker m-p100 \
--environment pytorch-0.3.1-gpu-py3 \
--config neptune.yaml \
--input /input/metadata.csv \
main.py 

```

You could have run it easily with both of those functions executed in the `main.py` :

```python
if __name__ == '__main__':
    prepare_metadata()
    train_evaluate_predict_cv()
```
but recalculating metadata every time you run your pipeline doesn't seem like a good idea :).

The model will be saved in the:

```yaml
  experiment_dir: /output/experiment
```

and the `submission.csv` will be saved in `/output/experiment/submission.csv`

You can easily use models trained during one experiment in other experiments.
For example when running evaluation we need to use the previous model folder in our experiment. We do that by:

changing `main.py` 

```python
  CLONE_EXPERIMENT_DIR_FROM = '/SAL-2/output/experiment'
```

and

```python
if __name__ == '__main__':
    evaluate_predict_cv()
```

and running the following command:


```bash
neptune send --worker m-p100 \
--environment pytorch-0.3.1-gpu-py3 \
--config neptune.yaml \
--input /input/metadata.csv \
--input /SAL-2 \
main.py
```

#### Local
Login to neptune if you want to use it
```bash
neptune account login
```

Prepare metadata
Change `main.py':
```python
if __name__ == '__main__':
    prepare_metadata()
```

run

```bash
neptune run --config neptune.yaml main.py prepare_metadata
```

Training and inference
Change `main.py':
```python
if __name__ == '__main__':
    train_evaluate_predict_cv()
```

```bash
neptune run --config neptune.yaml main.py
```

You can always run it with pure python :snake:

```bash
python main.py 
```

## References
1.Lovash Loss

```
@InProceedings{Berman_2018_CVPR,
author = {Berman, Maxim and Rannen Triki, Amal and Blaschko, Matthew B.},
title = {The Lovász-Softmax Loss: A Tractable Surrogate for the Optimization of the Intersection-Over-Union Measure in Neural Networks},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2018}
}
```


## Get involved
You are welcome to contribute your code and ideas to this open solution. To get started:
1. Check [competition project](https://github.com/neptune-ml/open-solution-salt-detection/projects/1) on GitHub to see what we are working on right now.
1. Express your interest in paticular task by writing comment in this task, or by creating new one with your fresh idea.
1. We will get back to you quickly in order to start working together.
1. Check [CONTRIBUTING](CONTRIBUTING.md) for some more information.

## User support
There are several ways to seek help:
1. Kaggle [discussion](https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/61949) is our primary way of communication.
1. Submit an [issue](https://github.com/minerva-ml/open-solution-salt-detection/issues) directly in this repo.

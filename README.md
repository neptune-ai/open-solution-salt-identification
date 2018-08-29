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

| link to code | CV | LB |
|:---:|:---:|:---:|
|[solution 1](https://github.com/neptune-ml/open-solution-salt-detection/tree/solution-1)|0.413|0.745|
|solution 2|0.794|0.798|
|solution 3|0.807|0.801|
|solution 4|0.802|0.809|

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

Prepare metadata. It only needs to be **done once**

```bash
neptune send --worker m-p100 \
--environment pytorch-0.3.1-gpu-py3 \
--config configs/neptune.yaml \
main.py prepare_metadata

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

Let's train the model by running:

```bash
neptune send --worker m-p100 \
--environment pytorch-0.3.1-gpu-py3 \
--config configs/neptune.yaml \
--input /input/metadata.csv \
main.py train --pipeline_name unet

```

The model will be saved in the:

```yaml
  experiment_dir: /output/experiment
```

So we when running evaluation we need to use this folder in our experiment. We do that by:

changing `neptune.yaml` 

```yaml
  clone_experiment_dir_from: '/SAL-2/output/experiment'
```
and running the following command:


```bash
neptune send --worker m-p100 \
--environment pytorch-0.3.1-gpu-py3 \
--config configs/neptune.yaml \
--input /input/metadata.csv \
--input /SAL-2 \
main.py evaluate_predict --pipeline_name unet

```

#### Local
Login to neptune if you want to use it
```bash
neptune account login
```

Prepare metadata

```bash
neptune run --config configs/neptune.yaml main.py prepare_metadata
```

Training

```bash
neptune run --config configs/neptune.yaml main.py train --pipeline_name unet
```

Inference

```bash
neptune run --config configs/neptune.yaml main.py evaluate_predict --pipeline_name unet
```

You can always run it with pure python :snake:

```bash
python main.py prepare_metadata
```

```bash
python main.py -- train--pipeline_name unet
```

```bash
python main.py -- evaluate_predict --pipeline_name unet
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

# TGS Salt Identification Challenge
: Open Solution
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/minerva-ml/open-solution-home-credit/blob/master/LICENSE)

This is an open solution to the [TGS Salt Identification Challenge
](https://www.kaggle.com/c/tgs-salt-identification-challenge).

## Our goals
We are building entirely open solution to this competition. Specifically:
1. **Learning from the process** - updates about new ideas, code and experiments is the best way to learn data science. Our activity is especially useful for people who wants to enter the competition, but lack appropriate experience.
1. Encourage more Kagglers to start working on this competition.
1. Deliver open source solution with no strings attached. Code is available on our [GitHub repository :computer:](https://github.com/neptune-ml/open-solution-salt-detection). This solution should establish solid benchmark, as well as provide good base for your custom ideas and experiments. We care about clean code :smiley:
1. We are opening our experiments as well: everybody can have **live preview** on our experiments, parameters, code, etc. Check: [TGS Salt Identification Challenge :chart_with_upwards_trend:](https://app.neptune.ml/neptune-ml/Salt-Detection) and screens below.

| Train and validation monitor :bar_chart: |
|:---|:---|
|[![training monitor](https://gist.githubusercontent.com/jakubczakon/cac72983726a970690ba7c33708e100b/raw/b45dd02b6643a3805db42ab51a62293a2940c0be/neptune_salt.png)|

## Disclaimer
In this open source solution you will find references to the [neptune.ml](https://neptune.ml). It is free platform for community Users, which we use daily to keep track of our experiments. Please note that using neptune.ml is not necessary to proceed with this solution. You may run it as plain Python script :snake:.

# How to start?
## Learn about our solutions
1. Check [Kaggle forum](TODO) and participate in the discussions.
1. Check our [Wiki pages :house_with_garden:](https://github.com/neptune-ml/open-solution-salt-detection/wiki), where we document our work. See solutions below:

| link to code | name | CV | LB | link to description |
|:---:|:---:|:---:|:---:|:---:|
|[solution 1](https://github.com/neptune-ml/open-solution-salt-detection/tree/solution-1)|*chestnut* :chestnut:|0.413|0.745|[Vanilla Unet](https://github.com/neptune-ml/open-solution-salt-detection/wiki/TODO)|
|solution 2|*four leaf clover* :four_leaf_clover:|0.794|0.798|[Deeper unet with smart augmentations](https://github.com/neptune-ml/open-solution-salt-detection/wiki/TODO)||

## Start experimenting with ready-to-use code
You can jump start your participation in the competition by using our starter pack. Installation instruction below will guide you through the setup.

### Installation *(fast track)*
1. Clone repository and install requirements (*use Python3.5*)
1. Register to the [neptune.ml](https://neptune.ml) _(if you wish to use it)_
1. Run experiment based on [unet](https://github.com/neptune-ml/open-solution-salt-detection/wiki/TODO):

:trident:
```bash
neptune account login
neptune run --config configs/neptune.yaml main.py train --pipeline_name unet
```

```bash
neptune account login
neptune run --config configs/neptune.yaml main.py evaluate_predict --pipeline_name unet
```

:snake:
```bash
python main.py -- train--pipeline_name unet
```

```bash
python main.py -- evaluate_predict --pipeline_name unet
```

### Installation *(step by step)*
[Step by step installation :desktop_computer:](https://github.com/neptune-ml/open-solution-salt-detection/wiki/Step-by-step-installation)

## Get involved
You are welcome to contribute your code and ideas to this open solution. To get started:
1. Check [competition project](https://github.com/minerva-ml/open-solution-home-credit/projects/1) on GitHub to see what we are working on right now.
1. Express your interest in paticular task by writing comment in this task, or by creating new one with your fresh idea.
1. We will get back to you quickly in order to start working together.
1. Check [CONTRIBUTING](CONTRIBUTING.md) for some more information.

## User support
There are several ways to seek help:
1. Kaggle [discussion](https://www.kaggle.com/c/TODO) is our primary way of communication.
1. Read project's [Wiki](https://github.com/neptune-ml/open-solution-salt-detection/wiki), where we publish descriptions about the code, pipelines and supporting tools such as [neptune.ml](https://neptune.ml).
1. Submit an [issue]((https://github.com/minerva-ml/open-solution-salt-detection/issues)) directly in this repo.

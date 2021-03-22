# TGS Salt Identification Challenge
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/minerva-ml/open-solution-home-credit/blob/master/LICENSE)

This is an open solution to the [TGS Salt Identification Challenge](https://www.kaggle.com/c/tgs-salt-identification-challenge).

# Note
**Unfortunately, we can no longer provide support for this repo. Hopefully, it should still work, but if it doesn't, we cannot really help.** 

## More competitions :sparkler:
Check collection of [public projects :gift:](https://ui.neptune.ai/-/explore), where you can find multiple Kaggle competitions with code, experiments and outputs.

## Our goals
We are building entirely open solution to this competition. Specifically:
1. **Learning from the process** - updates about new ideas, code and experiments is the best way to learn data science. Our activity is especially useful for people who wants to enter the competition, but lack appropriate experience.
1. Encourage more Kagglers to start working on this competition.
1. Deliver open source solution with no strings attached. Code is available on our [GitHub repository :computer:](https://github.com/neptune-ai/open-solution-salt-detection). This solution should establish solid benchmark, as well as provide good base for your custom ideas and experiments. We care about clean code :smiley:
1. We are opening our experiments as well: everybody can have **live preview** on our experiments, parameters, code, etc. Check: [TGS Salt Identification Challenge :chart_with_upwards_trend:](https://ui.neptune.ai/neptune-ai/Salt-Detection) or screen below.

|Train and validation monitor :bar_chart:|
|:---:|
|[![training monitor](https://gist.githubusercontent.com/jakubczakon/cac72983726a970690ba7c33708e100b/raw/b45dd02b6643a3805db42ab51a62293a2940c0be/neptune_salt.png)](https://ui.neptune.ai/-/dashboard/experiment/3dfce6cf-3031-4e9a-b95c-1ac8b5bb0026)|

## Disclaimer
In this open source solution you will find references to the [neptune.ai](https://neptune.ai). It is free platform for community Users, which we use daily to keep track of our experiments. Please note that using neptune.ai is not necessary to proceed with this solution. You may run it as plain Python script :snake:.

# How to start?
## Learn about our solutions
1. Check [Kaggle forum](https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/61949) and participate in the discussions.
1. See solutions below:

| Link to Experiments | CV | LB | Open |
|:---:|:---:|:---:|:---:|
|[solution 1](https://ui.neptune.ai/neptune-ai/Salt-Detection?namedFilterId=05e37f9f-c50c-4ba0-8065-92cd74eb9052)|0.413|0.745|True|
|[solution 2](https://ui.neptune.ai/neptune-ai/Salt-Detection?namedFilterId=57f36441-f0aa-4071-a05b-eb45fa0648e5)|0.794|0.798|True|
|[solution 3](https://ui.neptune.ai/neptune-ai/Salt-Detection?namedFilterId=c92051e6-97b6-40ba-b293-52fba301f9d7)|0.807|0.801|True|
|[solution 4](https://ui.neptune.ai/neptune-ai/Salt-Detection?namedFilterId=94881f72-46ad-4c84-829d-39e87c92937f)|0.802|0.809|True|
|[solution 5](https://ui.neptune.ai/neptune-ai/Salt-Detection?namedFilterId=60133d85-ab31-4395-b0e9-37deb25ecc94)|0.804|0.813|True|
|[solution 6](https://ui.neptune.ai/neptune-ai/Salt-Detection?namedFilterId=ab96e5df-3f1b-4516-9df0-4492e0199c71)|0.819|0.824|True|
|[solution 7](https://ui.neptune.ai/neptune-ai/Salt-Detection?namedFilterId=0810785e-ebab-4173-8e9e-8fe560095b77)|0.829|0.837|True|
|[solution 8](https://ui.neptune.ai/neptune-ai/Salt-Detection?namedFilterId=bda70048-f037-4c0d-a096-15ea93fd8924)|0.830|0.845|True|
|[solution 9](https://ui.neptune.ai/neptune-ai/Salt-Detection?namedFilterId=c21fc5a2-437a-412f-86e1-078fe31e025d)|0.853|0.849|True|


## Start experimenting with ready-to-use code
You can jump start your participation in the competition by using our starter pack. Installation instruction below will guide you through the setup.

### Installation 
#### Clone repository
```bash
git clone https://github.com/minerva-ml/open-solution-salt-identification.git
```
#### Set-up environment

You can setup the project with default env variables and open `NEPTUNE_API_TOKEN` by running:

```bash
source Makefile
```

I suggest at least reading the step-by-step instructions to know what is happening.

Install conda environment salt

```bash
conda env create -f environment.yml
```

After it is installed you can activate/deactivate it by running:

```bash
conda activate salt
```

```bash
conda deactivate
```

Register to the [neptune.ai](https://neptune.ai) _(if you wish to use it)_ even if you don't register you can still
see your experiment in Neptune. Just go to [shared/showroom project](https://ui.neptune.ai/o/shared/org/showroom/experiments) and find it.

Set environment variables `NEPTUNE_API_TOKEN` and `CONFIG_PATH`.

If you are using the default `neptune.yaml` config then run:
```bash
export export CONFIG_PATH=neptune.yaml
```

otherwise you can change to your config.

**Registered in Neptune**:

Set `NEPTUNE_API_TOKEN` variable with your personal token:

```bash
export NEPTUNE_API_TOKEN=your_account_token
```

Create new project in Neptune and go to your config file (`neptune.yaml`) and change `project` name:

```yaml
project: USER_NAME/PROJECT_NAME
``` 

**Not registered in Neptune**:

open token
```bash
export NEPTUNE_API_TOKEN=eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5tbCIsImFwaV9rZXkiOiJiNzA2YmM4Zi03NmY5LTRjMmUtOTM5ZC00YmEwMzZmOTMyZTQifQ==
```

#### Create data folder structure and set data paths in your config file (`neptune.yaml`)
Suggested directory structure:

```
project
|--   README.md
|-- ...
|-- data
    |-- raw
         |-- train 
            |-- images 
            |-- masks
         |-- test 
            |-- images
         |-- train.csv
         |-- sample_submission.csv
    |-- meta
        │-- depths.csv
        │-- metadata.csv # this is generated
        │-- auxiliary_metadata.csv # this is generated
    |-- stacking_data
        |-- out_of_folds_predictions # put oof predictions for multiple models/pipelines here
    |-- experiments
        |-- baseline # this is where your experiment files will be dumped
            |-- checkpoints # neural network checkpoints
            |-- transformers # serialized transformers after fitting
            |-- outputs # outputs of transformers if you specified save_output=True anywhere
            |-- out_of_fold_train_predictions.pkl # oof predictions on train
            |-- out_of_fold_test_predictions.pkl # oof predictions on test
            |-- submission.csv
        |-- empty_non_empty 
        |-- new_idea_exp 
```

in `neptune.yaml` config file change data paths if you decide on a different structure:

```yaml
  # Data Paths
  train_images_dir: data/raw/train
  test_images_dir: data/raw/test
  metadata_filepath: data/meta/metadata.csv
  depths_filepath: data/meta/depths.csv
  auxiliary_metadata_filepath: data/meta/auxiliary_metadata.csv
  stacking_data_dir: data/stacking_data
```

### Run experiment based on U-Net:

Prepare metadata:

```bash
python prepare_metadata.py
```

Training and inference.
Everything happens in `main.py`.
Whenever you try new idea make sure to change the name of the experiment:

```python
EXPERIMENT_NAME = 'baseline'
```

to a new name. 

```bash
python main.py
```

You can always change the pipeline you want ot run in the main.
For example, if I want to run just training and evaluation I can change  `main.py':
```python
if __name__ == '__main__':
    train_evaluate_cv()
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
1. Check [competition project](https://github.com/neptune-ai/open-solution-salt-detection/projects/1) on GitHub to see what we are working on right now.
1. Express your interest in paticular task by writing comment in this task, or by creating new one with your fresh idea.
1. We will get back to you quickly in order to start working together.
1. Check [CONTRIBUTING](CONTRIBUTING.md) for some more information.

## User support
There are several ways to seek help:
1. Kaggle [discussion](https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/61949) is our primary way of communication.
1. Submit an [issue](https://github.com/minerva-ml/open-solution-salt-detection/issues) directly in this repo.

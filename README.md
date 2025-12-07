# MIE1517-2025-Fall---Final-Report-Team20-

![Team20](./Team.png)
## 1. Intro
This github repository is for the submission of the final report of Team 20 in MIE1517 (2025 Fall) - Introduction to Deep Learning. 

## 2. Requirements
Run if you need to install required pakages.
```
pip install -r requirements.txt
```
For pytorch installation, please find a compatible version for your environment.

## 3. Downloading Audio Dataset
For this project, GTsinger audio dataset [Zhang et al., 2024](https://arxiv.org/pdf/2409.13832) for training our models. We processed the dataset, leaving only English version singing '.wav' files due to the limitation of memory and the training time. Please download the editted GTsinger zip file from [GTsinger_team20](https://drive.google.com/file/d/11aQMAexLnb_Qdb232ytvW-W8ImTae6BC/view?usp=sharing) and put the zip file in the `data` folder directory.

```
project/
 ├── README.md
 ├── data/
 │    ├── GTsinger.zip
```

## 4. Run
Open the 'Final_Report_Team20.ipynb' and walk through all cells to process dataset, train and test models.

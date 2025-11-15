# Project Setup Guide

## Dataset Download Instructions

Due to the large size of the datasets, they are not included in this repository. Please follow these steps to obtain the required data:

### Steps to Download Data

1. **Visit the Kaggle Competition Page**
   
   Navigate to: [Home Credit Default Risk Competition](https://www.kaggle.com/competitions/home-credit-default-risk/overview)

2. **Access the Data Section**
   
   Click on the "Data" tab in the competition navigation menu

3. **Download the Dataset**
   
   Download all the required data files from the competition page
   
   > **Note:** You may need to accept the competition rules and have a Kaggle account to download the data

4. **Place Data in Project Directory**
   
   Extract and place all downloaded files in the following directory:
```
   static/data/processData/
```

5. **Run the Notebooks**
   
   Once the data is in place, you can run the notebooks as needed

## Directory Structure

After downloading and placing the data, your directory structure should look like:
```
project-root/
├── static/
│   └── data/
│       └── processData/
│           ├── [downloaded data files]
│           └── ...
└── [notebooks and other project files]
```

## Requirements

- A Kaggle account (free to create)
- Acceptance of the competition rules
- Sufficient disk space for the datasets

## Questions?

If you encounter any issues with data download or setup, please open an issue in this repository.

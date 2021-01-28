import pandas as pd


def chooseDataset():
    print('[1] All of Canada')
    print('[2] Alberta')
    print('[3] British Columbia')
    print('[4] Manitoba')
    print('[5] New Brunswick')
    print('[6] Newfoundland and Labrador')
    print('[7] Northwest Territories')
    print('[8] Nova Scotia')
    print('[9] Nunavut')
    print('[10] Ontario')
    print('[11] Prince Edward Island')
    print('[12] Quebec')
    print('[13] Saskatchewan')
    print('[14] Yukon')
    data = 0
    title = ''
    choice_data = int(input("Please choose a dataset (1-14): "))
    if choice_data == 1:
        data = pd.read_csv('../DataPreprocessing/data_by_regions/Canada-2020-12-01.csv')
        title = 'Canada'
    elif choice_data == 2:
        data = pd.read_csv('../DataPreprocessing/data_by_regions/Alberta-2020-12-01.csv')
        title = 'Alberta'
    elif choice_data == 3:
        data = pd.read_csv('../DataPreprocessing/data_by_regions/British Columbia-2020-12-01.csv')
        title = 'British Columbia'
    elif choice_data == 4:
        data = pd.read_csv('../DataPreprocessing/data_by_regions/Manitoba-2020-12-01.csv')
        title = 'Manitoba'
    elif choice_data == 5:
        data = pd.read_csv('../DataPreprocessing/data_by_regions/New Brunswick-2020-12-01.csv')
        title = 'New Brunswick'
    elif choice_data == 6:
        data = pd.read_csv('../DataPreprocessing/data_by_regions/Newfoundland and Labrador-2020-12-01.csv')
        title = 'Newfoundland and Labrador'
    elif choice_data == 7:
        data = pd.read_csv('../DataPreprocessing/data_by_regions/Northwest Territories-2020-12-01.csv')
        title = 'Northwest Territories'
        print("\n-------SELECTED REGION LACKS DATA\n\n")
    elif choice_data == 8:
        data = pd.read_csv('../DataPreprocessing/data_by_regions/Nova Scotia-2020-12-01.csv')
        title = 'Nova Scotia'
    elif choice_data == 9:
        data = pd.read_csv('../DataPreprocessing/data_by_regions/Nunavut-2020-12-01.csv')
        title = 'Nunavut'
        print("\n-------SELECTED REGION LACKS DATA\n\n")
    elif choice_data == 10:
        data = pd.read_csv('../DataPreprocessing/data_by_regions/Ontario-2020-12-01.csv')
        title = 'Ontario'
    elif choice_data == 11:
        data = pd.read_csv('../DataPreprocessing/data_by_regions/Prince Edward Island-2020-12-01.csv')
        title = 'Prince Edward Island'
        print("\n-------SELECTED REGION LACKS DATA\n\n")
    elif choice_data == 12:
        data = pd.read_csv('../DataPreprocessing/data_by_regions/Quebec-2020-12-01.csv')
        title = 'Quebec'
    elif choice_data == 13:
        data = pd.read_csv('../DataPreprocessing/data_by_regions/Saskatchewan-2020-12-01.csv')
        title = 'Saskatchewan'
    elif choice_data == 14:
        data = pd.read_csv('../DataPreprocessing/data_by_regions/Yukon-2020-12-01.csv')
        title = 'Yukon'
        print("\n-------SELECTED REGION LACKS DATA\n\n")

    return data, title

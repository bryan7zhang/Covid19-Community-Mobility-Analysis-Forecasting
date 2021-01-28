import pandas as pd

# date of WHO declaration of COVID-19 Pandemic + Canada announcement
REFERENCE_DATE = pd.to_datetime('2020-03-11')

# 13 provinces + Canada
REGIONS = ['Canada', 'Alberta', 'British Columbia', 'Manitoba', 'New Brunswick',
           'Newfoundland and Labrador', 'Northwest Territories', 'Nova Scotia', 'Nunavut',
           'Ontario', 'Prince Edward Island', 'Quebec', 'Saskatchewan', 'Yukon']

# regional populations in Canada of first quarter of 2020
REGION_POPULATIONS = {'Canada': 37899277, 'Alberta': 4402045, 'British Columbia': 5131575,
                      'Manitoba': 1377004, 'New Brunswick': 780040, 'Newfoundland and Labrador': 523631,
                      'Northwest Territories': 45119, 'Nova Scotia': 975898, 'Nunavut': 38726,
                      'Ontario': 14689075, 'Prince Edward Island': 158629, 'Quebec': 8556650,
                      'Saskatchewan': 1179154, 'Yukon':41731}

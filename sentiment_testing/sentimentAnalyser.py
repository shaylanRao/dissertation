import json
import numpy as np
import pandas as pd

from IPython.display import display
from ibm_watson import ToneAnalyzerV3
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

authenticator = IAMAuthenticator('j6wBu5zuY4Kq2gyu0MGSXNg2Qc2Zsz-Hqye4mTVM-lZ2')
tone_analyzer = ToneAnalyzerV3(
    version='2017-09-21',
    authenticator=authenticator
)

tone_analyzer.set_service_url(
    'https://api.eu-gb.tone-analyzer.watson.cloud.ibm.com/instances/54ddd4d4-1449-40a7-8c05-fb9494afa611')

sample_text = "'Team, I know that times are tough! Product '\
    'sales have been disappointing for the past three '\
    'quarters. We have a competitive product, but we '\
    'need to do a better job of selling it!'"

column_names = ["anger", "fear", "joy", "sadness", "analytical", "confident", "tentative"]


def analyse(text):
    response = tone_analyzer.tone({'text': text},
                                  sentences=True
                                  ).get_result()
    return response


def format_for_analysis(raw_text):
    return raw_text


def array_maker(json_output):
    main_df = pd.DataFrame(columns=column_names)
    array = np.array("")
    for i in range(2):
        array = np.append(array, [json_output['sentences_tone'][i]['sentence_id']])
    return array


json_values = (analyse(format_for_analysis(sample_text)))
# print(array_maker(json_values))
main_df = pd.DataFrame(columns=column_names)
df2 = {'anger': '0.3242', 'sadness': '0.8864', 'analytical': '0.0234'}
main_df = main_df.append(df2, ignore_index=True)
display(main_df)

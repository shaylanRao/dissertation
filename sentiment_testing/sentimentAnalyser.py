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


def get_senti(text):
    if text == "":
        return None
    response = tone_analyzer.tone({'text': text},
                                  sentences=True
                                  ).get_result()
    try:
        analysis = response['sentences_tone'][0]['tones']
        if analysis:
            for item in analysis:
                print(item)
        else:
            return "No Tone"
    # only one sentence (the next tweet is a song)
    except KeyError:
        try:
            return response['document_tone']['tones'][0]
        except IndexError:
            return "Gibberish"
    return "---------------"


def format_for_analysis(raw_text):
    return raw_text


def array_maker(json_output):
    main_df = pd.DataFrame(columns=column_names)
    array = np.array("")
    for i in range(2):
        array = np.append(array, [json_output['sentences_tone'][i]['sentence_id']])
    return array


def gain_tone_values(text):
    if text == "":
        return None
    else:
        try:
            senti_json = get_senti(text)
            document_tone = senti_json['document_tone']['tones'][0]
            # print(document_tone['tone_id'])
            print(senti_json[0])
        except IndexError:
            print("No tone")


def main():
    json_values = (get_senti(format_for_analysis(sample_text)))
    # print(array_maker(json_values))
    main_df = pd.DataFrame(columns=column_names)
    df2 = {'anger': '0.3242', 'sadness': '0.8864', 'analytical': '0.0234'}
    display(main_df)
    main_df = main_df.append(df2, ignore_index=True)


# test_text = "I am so happy, i am scared"
# print(test_text)
# gain_tone_values(test_text)
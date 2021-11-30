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

sample_text_old = "Team, I know that times are tough! \n Product sales have been disappointing for the past three quarters. \n We have a competitive product, but we need more!"

sample_text = "Donda is a work of art \n They said I was mad at the Grammys BUT IM LOOKING AT MY GRAMMY RN \n This isnt enough I need 4K"

column_names = ["anger", "fear", "joy", "sadness", "analytical", "confident", "tentative"]


def get_senti(text):
    main_df = pd.DataFrame(columns=column_names)
    # if parameter us empty
    if text == "":
        return None
    # Analyse the text (all sentences)
    response = tone_analyzer.tone({'text': text},
                                  sentences=True
                                  ).get_result()
    # get tones for each sentence (if multiple sentences)
    try:
        analysis = response['sentences_tone']
        # print("MULTIPLE SENTENCES")
        for item in analysis:
            df2 = sentence_analyser(item['tones'])
            main_df = main_df.append(df2, ignore_index=True)                # append tone values to total dataframe
    # only one sentence (the next tweet is a song)
    except KeyError:
        # Returns the sentiment score value for the single sentence
        try:
            df = sentence_analyser(response['document_tone']['tones'])
            main_df = main_df.append(df, ignore_index=True)
            return main_df.fillna(0).mean()
        # No tone identified
        except IndexError:
            return "Gibberish"
    # return dataframe from multiple sentences
    return main_df.fillna(0).mean()


#
def array_maker(json_output):
    main_df = pd.DataFrame(columns=column_names)
    array = np.array("")
    for i in range(2):
        array = np.append(array, [json_output['sentences_tone'][i]['sentence_id']])
    return array


def sentence_analyser(item):
    if not item:
        return None
    # For each type of tone in a sentence (usually just one)
    tone_id_list = []
    tone_value = []
    for aspect in item:
        tone_id_list.append(aspect['tone_id'])
        tone_value.append(aspect['score'])

    df2 = dict(zip(tone_id_list, tone_value))
    return df2


def main():
    # json_values = (get_senti(sample_text))
    data = []
    main_df = pd.DataFrame(columns=column_names)
    df2 = {'anger': '0.3242', 'sadness': '0.8864', 'analytical': '0.0234'}
    main_df = main_df.append(df2, ignore_index=True)

    name = ['fear', 'sadness', 'joy']
    values = ['0.232', '0.342', '0.435']
    df2 = dict(zip(name, values))
    # df2 = {name[0]: '0.453', 'sadness': '0.233', 'joy': '0.324'}
    main_df = main_df.append(df2, ignore_index=True)
    display(main_df)


# main()
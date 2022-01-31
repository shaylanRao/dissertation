import pandas as pd
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson import ToneAnalyzerV3

AUTH_CODE = 'j6wBu5zuY4Kq2gyu0MGSXNg2Qc2Zsz-Hqye4mTVM-lZ2'
VERSION = '2017-09-21'
SERVICE_URL = 'https://api.eu-gb.tone-analyzer.watson.cloud.ibm.com/instances/54ddd4d4-1449-40a7-8c05-fb9494afa611'
COLUMN_HEADINGS = ["anger", "fear", "joy", "sadness", "analytical", "confident", "tentative"]
COLUMN_HEADINGS_LYRICS = ["lyric_{0}".format(i) for i in COLUMN_HEADINGS]

authenticator = IAMAuthenticator(AUTH_CODE)
tone_analyzer = ToneAnalyzerV3(
    version=VERSION,
    authenticator=authenticator
)
tone_analyzer.set_service_url(SERVICE_URL)


def sentence_analyser(item):
    if not item:
        return None
    tone_id_list = []
    tone_value = []
    # For each type of tone in a sentence (usually just one)
    for aspect in item:
        tone_id_list.append(aspect['tone_id'])
        tone_value.append(aspect['score'])

    # makes a single record to add to dataframe
    df2 = dict(zip(tone_id_list, tone_value))
    return df2


def get_text_senti(text):
    print(text)
    label_df = pd.DataFrame(columns=COLUMN_HEADINGS)
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
            label_df = label_df.append(df2, ignore_index=True)  # append tone values to total dataframe

            # return dataframe from multiple sentences
        label_df = label_df.fillna(0).mean()
        return label_df
    # only one sentence (the next tweet is a song)
    except KeyError:
        # Returns the sentiment score value for the single sentence
        try:
            df = sentence_analyser(response['document_tone']['tones'])
            label_df = label_df.append(df, ignore_index=True)
            label_df = label_df.fillna(0).mean()
            return label_df
        # No tone identified
        except IndexError:
            return "Gibberish"


def get_l_senti(lyrics):
    label_df = pd.DataFrame(columns=COLUMN_HEADINGS)
    # if parameter us empty
    if lyrics == "":
        return None
    # Analyse the text (all sentences)
    response = tone_analyzer.tone({'text': lyrics},
                                  sentences=True
                                  ).get_result()

    df = sentence_analyser(response['document_tone']['tones'])
    label_df = label_df.append(df, ignore_index=True)
    label_df = label_df.fillna(0)
    # Renames column headings
    label_df.columns = COLUMN_HEADINGS_LYRICS
    return label_df

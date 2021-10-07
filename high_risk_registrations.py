#install missing libraries after update
# !pip install boto3
# !pip install delorean
# !pip install bs4
# !pip install zenpy
# !pip install stripe
# !pip install slack
# !pip install slacker
# !pip install fastparquet
# !pip install plotly
# !pip install jupyterthemes
# !pip install pandasql
# !pip install gspread
# !pip install gspread_dataframe
# !pip install spam_lists
# !pip install nltk
# !pip install shap

from my_set_up import *
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import pandas as pd
import itertools
import spam_lists
pd.set_option('display.float_format', '{:.2f}'.format)
# pd.options.display.precision = 2
# pd.options.display.max_rows = 10


# Gibberish Detecting Functions
"""
Gibberish Classifier
Author: ProgramFOX
Licensed under CPOL v1.02: http://www.codeproject.com/info/cpol10.aspx
"""

# from __future__ import division
import re
import math

def split_in_chunks(text, chunk_size):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])
    if len(chunks) > 1 and len(chunks[-1]) < 10:
        chunks[-2] += chunks[-1]
        chunks.pop(-1)
    return chunks


def unique_chars_per_chunk_percentage(text, chunk_size):
    chunks = split_in_chunks(text, chunk_size)
    unique_chars_percentages = []
    for chunk in chunks:
        total = len(chunk)
        unique = len(set(chunk))
        unique_chars_percentages.append(unique / total)
    return sum(unique_chars_percentages) / len(unique_chars_percentages) * 100


def vowels_percentage(text):
    vowels = 0
    total = 0
    for c in text:
        if not c.isalpha():
            continue
        total += 1
        if c in "aeiouAEIOU":
            vowels += 1
    if total != 0:
        return vowels / total * 100
    else:
        return 0


def word_to_char_ratio(text):
    chars = len(text)
    words = len([x for x in re.split(r"[\W_]", text) if x.strip() != ""])
    return words / chars * 100


def deviation_score(percentage, lower_bound, upper_bound):
    if percentage < lower_bound:
        return math.log(lower_bound - percentage, lower_bound) * 100
    elif percentage > upper_bound:
        return math.log(percentage - upper_bound, 100 - upper_bound) * 100
    else:
        return 0


def classify(text):
    if text is None or len(text) == 0:
        return 0.0
    ucpcp = unique_chars_per_chunk_percentage(text, 35)
    vp = vowels_percentage(text)
    wtcr = word_to_char_ratio(text)

    ucpcp_dev = max(deviation_score(ucpcp, 45, 50), 1)
    vp_dev = max(deviation_score(vp, 35, 45), 1)
    wtcr_dev = max(deviation_score(wtcr, 15, 20), 1)

    return max((math.log10(ucpcp_dev) + math.log10(vp_dev) +
                math.log10(wtcr_dev)) / 6 * 100, 1)


# Strangeness text function
# import nltk
# # nltk.download('brown')

# from nltk.corpus import brown
# from collections import Counter
# import numpy as np

# text = '\n  '.join([' '.join([w for w in s]) for s in brown.sents()])

# unigrams = Counter(text)
# bigrams = Counter(text[i:(i+2)] for i in range(len(text)-2))
# trigrams = Counter(text[i:(i+3)] for i in range(len(text)-3))

# weights = [0.001, 0.01, 0.989]

# def strangeness(text):
#     r = 0
#     text = '  ' + text + '\n'
#     for i in range(2, len(text)):
#         char = text[i]
#         context1 = text[(i-1):i]
#         context2 = text[(i-2):i]
#         num = unigrams[char] * weights[0] + bigrams[context1+char] * weights[1] + trigrams[context2+char] * weights[2] 
#         den = sum(unigrams.values()) * weights[0] + unigrams[context1] * weights[1] + bigrams[context2] * weights[2]
#         r -= np.log(num / den)
#     return r / (len(text) - 2)


#fraud model
start_time = datetime.now()
print('Start Time: ', start_time)

query = """
DECLARE numdays INT64 DEFAULT 90;
DECLARE numhours INT64 DEFAULT 1;

with data as (
    select
        case when d.risk_scores < -50 then -50
            when d.risk_scores between -26 and -50 then -25
            when d.risk_scores between -25 and 0 then 0
            when d.risk_scores between 0 and 25 then 25
            when d.risk_scores between 26 and 50 then 50
            when d.risk_scores between 51 and 75 then 75
            when d.risk_scores between 76 and 100 then 100
            when d.risk_scores between 101 and 125 then 125
            when d.risk_scores between 126 and 150 then 150
            when d.risk_scores between 151 and 175 then 175
            when d.risk_scores between 176 and 200 then 200
            when d.risk_scores between 201 and 225 then 225
            when d.risk_scores between 226 and 250 then 250
            when d.risk_scores between 251 and 275 then 275
            when d.risk_scores between 276 and 300 then 300
            when d.risk_scores between 301 and 325 then 325
            when d.risk_scores between 326 and 350 then 350
            when d.risk_scores between 351 and 375 then 375
            when d.risk_scores between 376 and 400 then 400
            when d.risk_scores between 401 and 425 then 425
            when d.risk_scores between 426 and 450 then 450
            when d.risk_scores between 451 and 475 then 475
            when d.risk_scores between 476 and 500 then 500
            when d.risk_scores > 500 then 501
        else null end as risk_score --was risk_score_bin
        ,d.*
    from (
        select
            --utilize rule accuracy within the final risk_scores
            case when c.final_score != c.score then c.score - c.final_score else c.score end as risk_scores
            ,(case when c.prev_blocked_email_handle = 1 then 'rule_prev_blocked_email_handle' else '' end||','
                ||case when c.uncommon_domain = 1 then 'rule_uncommon_domain' else '' end||','
                ||case when c.disposable_domain = 1 then 'rule_disposable_domain' else '' end||','
                ||case when c.bot_score_bucket > 90 then 'rule_bot_score_bucket' else '' end||','
                ||case when c.is_emulator = 1 then 'rule_is_emulator' else '' end||','
                ||case when lower(c.origin) like 'web%'
                        or lower(c.origin) = 'ipod'
                        or lower(c.origin) = 'ipad'
                        or lower(c.origin) = 'unknown' 
                    then 'rule_origin' else '' end||','
                ||case when c.likely_invalid_email = 1 then 'rule_likely_invalid_email' else '' end||','
                ||case when c.zip in ('32578','8260','34292','28097','32202','67108','37938','46617'
                                        ,'46601','85034','26181','30303','32327','45458','95119','39203')
                    then 'rule_zip' else '' end||','
                ||case when c.signup_method = 'email' then 'rule_signup_method' else '' end||','
                ||case when c.registration_hour in (0,17,18,19,20,21,22,23) then 'rule_registration_hour' else '' end||','
                ||case when c.email_nums in ('63','82','38','55','43','44','97','56','36','93','85','59','31',
                        '29','99','754','40','21','74','33','60','27','65','45','95','50','79','90','76','78','32',
                        '19','68','87','2','26','42','49','64','456','73','71','732','67','88','61','72','643','89',
                        '76,56','46','15','96','54','34','66','83','48','18','84','35','91','12','69','98','8')
                    then 'rule_email_nums' else '' end||','
                ||case when c.count_email_nums > 5 then 'rule_count_email_nums' else '' end||','
                ||case when c.state_name in ('VA','NJ','OH','ME','NV','WI','IN','MA','WY','SC','DE','UT','CO',
                                                'AR','MO','ND','KY','IA','HI','MD','NH','SD','AL','NE','RI','NM')
                    then 'rule_state_name' else '' end||','
                ||case when c.emails_are_bouncing = 1 then 'rule_emails_are_bouncing' else '' end||','
                ||case when c.is_fraud_device_id = 1 then 'rule_is_fraud_device_id' else '' end||','
                ||case when c.is_fraud_ip_address = 1 then 'rule_is_fraud_ip_address' else '' end||','
                ||case when c.bot_score_bucket > 60 
                            and c.registration_hour in (0,2,4,5,6,6,8,11,12,14,14,14,15,15,17,19,19,20,21)
                    then 'rule_fraud_ip_reg_hour' else '' end||','
                ||case when c.state_name in ('MS','NE','IN','SC','AL','MS','DE','OK','AR','MN','NE','NE','NH','AL','NH','KS','AL','MN') 
                            and c.registration_hour in (4,16,10,7,6,14,12,9,5,6,4,23,12,9,13,4,11,10)
                        then 'rule_state_name_reg_hour' else '' end||','
                ||case when lower(c.origin) = 'iphone' and c.is_fraud_ip_address = 0 then 'rule_origin_fraud_ip' else '' end||','
                ||case when c.rules_triggered in (
                        ',,,,,origin,,,,,,,,,,is_fraud_ip_address,',',,,,,origin,,,,,email_nums,,,,,,',',,,bot_score_bucket,,origin,,,,,email_nums,,,,,is_fraud_ip_address,'
                        ,',,,,,origin,,,,registration_hour,,,,,,is_fraud_ip_address,',',,,,,origin,,,signup_method,,,,,,,is_fraud_ip_address,'
                        ,',,,,,origin,,,signup_method,registration_hour,,,,,,is_fraud_ip_address,',',,,bot_score_bucket,,,,zip,,,,,state_name,,,,'
                        ,',,,bot_score_bucket,,origin,,,,,,,,,,is_fraud_ip_address,',',,,bot_score_bucket,,origin,,,,,,,state_name,,,is_fraud_ip_address,'
                        ,',,,bot_score_bucket,,origin,,,,,,count_email_nums,,,,,',',,,bot_score_bucket,,origin,,,,,email_nums,,,,,,'
                        ,',,,bot_score_bucket,,origin,,,,,email_nums,,state_name,,,,',',,,bot_score_bucket,,origin,,,,registration_hour,,,,,,is_fraud_ip_address,'
                        ,',,,bot_score_bucket,,origin,,,,registration_hour,,,state_name,,,is_fraud_ip_address,',',,,bot_score_bucket,,origin,,,,registration_hour,,count_email_nums,,,,,'
                        ,',,,bot_score_bucket,,origin,,,,registration_hour,,count_email_nums,,,,is_fraud_ip_address,',',,,bot_score_bucket,,origin,,,,registration_hour,email_nums,,,,,,'
                        ,',,,bot_score_bucket,,origin,,,,registration_hour,email_nums,,,,,is_fraud_ip_address,',',uncommon_domain,,,,origin,,,signup_method,,,,state_name,,,is_fraud_ip_address,'
                        ,',,,bot_score_bucket,,origin,,,,registration_hour,email_nums,,state_name,,,,',',,,bot_score_bucket,,origin,,,signup_method,,,,,,,is_fraud_ip_address,'
                        ,',,,bot_score_bucket,,origin,,,signup_method,,,,state_name,,,is_fraud_ip_address,',',,,bot_score_bucket,,origin,,,signup_method,,email_nums,,,,,,'
                        ,',,,bot_score_bucket,,origin,,,signup_method,,email_nums,,,,,is_fraud_ip_address,',',,,bot_score_bucket,,origin,,,signup_method,registration_hour,email_nums,,,,,,'
                        ,',,,bot_score_bucket,,origin,,,signup_method,registration_hour,,,,,,is_fraud_ip_address,',',uncommon_domain,,bot_score_bucket,,origin,,,signup_method,registration_hour,,,,,,,'
                        ,',,,bot_score_bucket,,origin,,,signup_method,registration_hour,email_nums,,,,,is_fraud_ip_address,',',,,bot_score_bucket,,origin,,zip,,,,,,,,is_fraud_ip_address,'
                        ,',,,bot_score_bucket,,origin,,zip,,registration_hour,,,,,,is_fraud_ip_address,',',uncommon_domain,,,,origin,,,signup_method,,,,,,,is_fraud_ip_address,'
                        ,',uncommon_domain,,,,origin,,,signup_method,,,,,,is_fraud_device_id,is_fraud_ip_address,',',uncommon_domain,,,,origin,,,signup_method,registration_hour,,,,,,is_fraud_ip_address,'
                        ,',uncommon_domain,,,,origin,,,signup_method,,,,state_name,,is_fraud_device_id,is_fraud_ip_address,'
                        ,',uncommon_domain,,,,origin,,,signup_method,registration_hour,,,,,is_fraud_device_id,is_fraud_ip_address,'
                        ,',uncommon_domain,,,,origin,,zip,signup_method,,,,,,,is_fraud_ip_address,',',uncommon_domain,,,,origin,,zip,signup_method,registration_hour,,,,,,is_fraud_ip_address,'
                        ,',uncommon_domain,,bot_score_bucket,,origin,,,,,,,,,,,',',uncommon_domain,,bot_score_bucket,,origin,,,,,,,state_name,,,,'
                        ,',uncommon_domain,,bot_score_bucket,,origin,,,,registration_hour,,,,,,,',',uncommon_domain,,bot_score_bucket,,origin,,,signup_method,,,,,,,,'
                        ,',uncommon_domain,,bot_score_bucket,,origin,,,signup_method,,,,,,,is_fraud_ip_address,',',uncommon_domain,,bot_score_bucket,,origin,,,signup_method,,,,state_name,,,is_fraud_ip_address,'
                        ,',uncommon_domain,,bot_score_bucket,,origin,,,signup_method,,,count_email_nums,,,,is_fraud_ip_address,'
                        ,',uncommon_domain,,bot_score_bucket,,origin,,,signup_method,,,count_email_nums,state_name,,,is_fraud_ip_address,'
                        ,',uncommon_domain,,bot_score_bucket,,origin,,,signup_method,,email_nums,,,,,is_fraud_ip_address,',',uncommon_domain,,bot_score_bucket,,origin,,,signup_method,registration_hour,,,state_name,,,is_f'
                        ,',uncommon_domain,,bot_score_bucket,,origin,,,signup_method,registration_hour,,,,,,is_fraud_ip_address,'
                        ,',uncommon_domain,,bot_score_bucket,,origin,,,signup_method,registration_hour,,,state_name,,is_fr_fraud_ip_address,'
                        ,',uncommon_domain,,bot_score_bucket,,origin,,,signup_method,registration_hour,,count_email_nums,,dress,'
                        ,',uncommon_domain,,bot_score_bucket,,origin,,,signup_method,registration_hour,,count_email_nums,sraud_ip_address,'
                        ,',uncommon_domain,,bot_score_bucket,,origin,,zip,signup_method,,,,,,,is_fraud_ip_address,',',uncommon_domain,,,,origin,,,signup_method,registration_hour,,,state_name,,,is_fraud_ip_address,'
                        ,',uncommon_domain,,bot_score_bucket,,origin,,zip,signup_method,,email_nums,,,,,is_fraud_ip_address,'
                        ,',uncommon_domain,,bot_score_bucket,,origin,,zip,signup_method,registration_hour,,,,,,is_fraud_ip_address,'
                        ,',uncommon_domain,,bot_score_bucket,,origin,,zip,signup_method,registration_hour,,count_email_num_address,'
                        )
                    then 'rule_low_risk_rules_triggered' else '' end||','
                ||case when c.first_name in ('Author Pinor','Lucas Betty','Diane','Williams Sandy','Clinton','Williams'
                                            ,'Clinton Dean','Author','Woolley','Josephine','Marisa','Paula Ross','debbie')
                    then 'rule_risky_first_name' else '' end||','
                ||case when c.is_fraud_device_id = 1 
                        and c.bot_score_bucket > 60
                        and c.registration_hour in (0,2,4,5,6,6,8,11,12,14,14,14,15,15,17,19,19,20,21)
                    then 'fraud_dev_bot_reg_hr' else '' end||','
                ||case when c.count_accounts >= 15 then 'device_linked_ge_15_accts' else '' end||','
            ) as risk_rules_triggered
            ,c.*
        from (
            select
                b.*

                --low risk rule combos
                ,case when b.rules_triggered in 
                        (
                            ',,,,,origin,,,,,,,,,,is_fraud_ip_address,'
                            ,',,,,,origin,,,,,email_nums,,,,,,'
                            ,',,,,,origin,,,,registration_hour,,,,,,is_fraud_ip_address,'
                            ,',,,,,origin,,,signup_method,,,,,,,is_fraud_ip_address,'
                            ,',,,,,origin,,,signup_method,registration_hour,,,,,,is_fraud_ip_address,'
                            ,',,,bot_score_bucket,,,,zip,,,,,state_name,,,,'
                            ,',,,bot_score_bucket,,origin,,,,,,,,,,is_fraud_ip_address,'
                            ,',,,bot_score_bucket,,origin,,,,,,,state_name,,,is_fraud_ip_address,'
                            ,',,,bot_score_bucket,,origin,,,,,,count_email_nums,,,,,'
                            ,',,,bot_score_bucket,,origin,,,,,email_nums,,,,,,'
                            ,',,,bot_score_bucket,,origin,,,,,email_nums,,,,,is_fraud_ip_address,'
                            ,',,,bot_score_bucket,,origin,,,,,email_nums,,state_name,,,,'
                            ,',,,bot_score_bucket,,origin,,,,registration_hour,,,,,,is_fraud_ip_address,'
                            ,',,,bot_score_bucket,,origin,,,,registration_hour,,,state_name,,,is_fraud_ip_address,'
                            ,',,,bot_score_bucket,,origin,,,,registration_hour,,count_email_nums,,,,,'
                            ,',,,bot_score_bucket,,origin,,,,registration_hour,,count_email_nums,,,,is_fraud_ip_address,'
                            ,',,,bot_score_bucket,,origin,,,,registration_hour,email_nums,,,,,,'
                            ,',,,bot_score_bucket,,origin,,,,registration_hour,email_nums,,,,,is_fraud_ip_address,'
                            ,',,,bot_score_bucket,,origin,,,,registration_hour,email_nums,,state_name,,,,'
                            ,',,,bot_score_bucket,,origin,,,signup_method,,,,,,,is_fraud_ip_address,'
                            ,',,,bot_score_bucket,,origin,,,signup_method,,,,state_name,,,is_fraud_ip_address,'
                            ,',,,bot_score_bucket,,origin,,,signup_method,,email_nums,,,,,,'
                            ,',,,bot_score_bucket,,origin,,,signup_method,,email_nums,,,,,is_fraud_ip_address,'
                            ,',,,bot_score_bucket,,origin,,,signup_method,registration_hour,,,,,,is_fraud_ip_address,'
                            ,',,,bot_score_bucket,,origin,,,signup_method,registration_hour,email_nums,,,,,,'
                            ,',,,bot_score_bucket,,origin,,,signup_method,registration_hour,email_nums,,,,,is_fraud_ip_address,'
                            ,',,,bot_score_bucket,,origin,,zip,,,,,,,,is_fraud_ip_address,'
                            ,',,,bot_score_bucket,,origin,,zip,,registration_hour,,,,,,is_fraud_ip_address,'
                            ,',uncommon_domain,,,,origin,,,signup_method,,,,,,,is_fraud_ip_address,'
                            ,',uncommon_domain,,,,origin,,,signup_method,,,,,,is_fraud_device_id,is_fraud_ip_address,'
                            ,',uncommon_domain,,,,origin,,,signup_method,,,,state_name,,,is_fraud_ip_address,'
                            ,',uncommon_domain,,,,origin,,,signup_method,,,,state_name,,is_fraud_device_id,is_fraud_ip_address,'
                            ,',uncommon_domain,,,,origin,,,signup_method,registration_hour,,,,,,is_fraud_ip_address,'
                            ,',uncommon_domain,,,,origin,,,signup_method,registration_hour,,,,,is_fraud_device_id,is_fraud_ip_address,'
                            ,',uncommon_domain,,,,origin,,,signup_method,registration_hour,,,state_name,,,is_fraud_ip_address,'
                            ,',uncommon_domain,,,,origin,,zip,signup_method,,,,,,,is_fraud_ip_address,'
                            ,',uncommon_domain,,,,origin,,zip,signup_method,registration_hour,,,,,,is_fraud_ip_address,'
                            ,',uncommon_domain,,bot_score_bucket,,origin,,,,,,,,,,,'
                            ,',uncommon_domain,,bot_score_bucket,,origin,,,,,,,state_name,,,,'
                            ,',uncommon_domain,,bot_score_bucket,,origin,,,,registration_hour,,,,,,,'
                            ,',uncommon_domain,,bot_score_bucket,,origin,,,signup_method,,,,,,,,'
                            ,',uncommon_domain,,bot_score_bucket,,origin,,,signup_method,,,,,,,is_fraud_ip_address,'
                            ,',uncommon_domain,,bot_score_bucket,,origin,,,signup_method,,,,state_name,,,is_fraud_ip_address,'
                            ,',uncommon_domain,,bot_score_bucket,,origin,,,signup_method,,,count_email_nums,,,,is_fraud_ip_address,'
                            ,',uncommon_domain,,bot_score_bucket,,origin,,,signup_method,,,count_email_nums,state_name,,,is_fraud_ip_address,'
                            ,',uncommon_domain,,bot_score_bucket,,origin,,,signup_method,,email_nums,,,,,is_fraud_ip_address,'
                            ,',uncommon_domain,,bot_score_bucket,,origin,,,signup_method,registration_hour,,,,,,,'
                            ,',uncommon_domain,,bot_score_bucket,,origin,,,signup_method,registration_hour,,,,,,is_fraud_ip_address,'
                            ,',uncommon_domain,,bot_score_bucket,,origin,,,signup_method,registration_hour,,,state_name,,,is_fraud_ip_address,'
                            ,',uncommon_domain,,bot_score_bucket,,origin,,,signup_method,registration_hour,,,state_name,,is_fraud_device_id,is_fraud_ip_address,'
                            ,',uncommon_domain,,bot_score_bucket,,origin,,,signup_method,registration_hour,,count_email_nums,,,,is_fraud_ip_address,'
                            ,',uncommon_domain,,bot_score_bucket,,origin,,,signup_method,registration_hour,,count_email_nums,state_name,,,is_fraud_ip_address,'
                            ,',uncommon_domain,,bot_score_bucket,,origin,,zip,signup_method,,,,,,,is_fraud_ip_address,'
                            ,',uncommon_domain,,bot_score_bucket,,origin,,zip,signup_method,,email_nums,,,,,is_fraud_ip_address,'
                            ,',uncommon_domain,,bot_score_bucket,,origin,,zip,signup_method,registration_hour,,,,,,is_fraud_ip_address,'
                            ,',uncommon_domain,,bot_score_bucket,,origin,,zip,signup_method,registration_hour,,count_email_nums,,,,is_fraud_ip_address,'
                        )
                    then b.score -200 else 0 end as final_score
                from (
                select
                    case when a.score < -50 then 'lt-50'
                        when a.score between -26 and -50 then 'n25-n50'
                        when a.score between -25 and 0 then '0-n25'
                        when a.score between 0 and 25 then '0-25'
                        when a.score between 26 and 50 then '26-50'
                        when a.score between 51 and 75 then '51-75'
                        when a.score between 76 and 100 then '76-100'
                        when a.score between 101 and 125 then '101-125'
                        when a.score between 126 and 150 then '126-150'
                        when a.score between 151 and 175 then '151-175'
                        when a.score between 176 and 200 then '176-200'
                        when a.score between 201 and 225 then '201-225'
                        when a.score between 226 and 250 then '226-250'
                        when a.score between 251 and 275 then '251-275'
                        when a.score between 276 and 300 then '276-300'
                        when a.score between 301 and 325 then '301-325'
                        when a.score between 326 and 350 then '326-350'
                        when a.score between 351 and 375 then '351-375'
                        when a.score between 376 and 400 then '376-400'
                        when a.score between 401 and 425 then '401-425'
                        when a.score between 426 and 450 then '426-450'
                        when a.score between 451 and 475 then '451-475'
                        when a.score between 476 and 500 then '476-500'
                        when a.score > 501 then 'gt501'
                    else null end as score_bin 
                    --score rule names
                    ,(case when a.prev_blocked_email_handle = 1 then 'prev_blocked_email_handle' else '' end||','
                        ||case when a.uncommon_domain = 1 then 'uncommon_domain' else '' end||','
                        ||case when a.disposable_domain = 1 then 'disposable_domain' else '' end||','
                        ||case when a.bot_score_bucket > 90 then 'bot_score_bucket' else '' end||','
                        ||case when a.is_emulator = 1 then 'is_emulator' else '' end||','
                        ||case when lower(a.origin) like 'web%'
                                or lower(a.origin) = 'ipod'
                                or lower(a.origin) = 'ipad'
                                or lower(a.origin) = 'unknown' 
                            then 'origin' else '' end||','
                        ||case when a.likely_invalid_email = 1 then 'likely_invalid_email' else '' end||','
                        ||case when a.zip in ('32578','8260','34292','28097','32202','67108','37938','46617'
                                                ,'46601','85034','26181','30303','32327','45458','95119','39203')
                            then 'zip' else '' end||','
                        ||case when a.signup_method = 'email' then 'signup_method' else '' end||','
                        ||case when a.registration_hour in (0,17,18,19,20,21,22,23) then 'registration_hour' else '' end||','
                        ||case when a.email_nums in ('63','82','38','55','43','44','97','56','36','93','85','59','31',
                                '29','99','754','40','21','74','33','60','27','65','45','95','50','79','90','76','78','32',
                                '19','68','87','2','26','42','49','64','456','73','71','732','67','88','61','72','643','89',
                                '76,56','46','15','96','54','34','66','83','48','18','84','35','91','12','69','98','8')
                            then 'email_nums' else '' end||','
                        ||case when a.count_email_nums > 5 then 'count_email_nums' else '' end||','
                        ||case when a.state_name in ('VA','NJ','OH','ME','NV','WI','IN','MA','WY','SC','DE','UT','CO',
                                                        'AR','MO','ND','KY','IA','HI','MD','NH','SD','AL','NE','RI','NM')
                            then 'state_name' else '' end||','
                        ||case when a.emails_are_bouncing = 1 then 'emails_are_bouncing' else '' end||','
                        ||case when a.is_fraud_device_id = 1 then 'is_fraud_device_id' else '' end||','
                        ||case when a.is_fraud_ip_address = 1 then 'is_fraud_ip_address' else '' end||','
                        ||case when a.bot_score_bucket > 60 
                                    and a.registration_hour in (0,2,4,5,6,6,8,11,12,14,14,14,15,15,17,19,19,20,21)
                            then 'fraud_ip_reg_hour' else '' end||','
                        ||case when a.state_name in ('MS','NE','IN','SC','AL','MS','DE','OK','AR','MN','NE','NE','NH','AL','NH','KS','AL','MN') 
                                    and a.registration_hour in (4,16,10,7,6,14,12,9,5,6,4,23,12,9,13,4,11,10)
                            then 'state_name_reg_hour' else '' end||','
                        ||case when lower(a.origin) = 'iphone' and a.is_fraud_ip_address = 0 then 'origin_fraud_ip' else '' end||','
                        ||case when a.first_name in ('Author Pinor','Lucas Betty','Diane','Williams Sandy','Clinton','Williams'
                                                        ,'Clinton Dean','Author','Woolley','Josephine','Marisa','Paula Ross','debbie')
                            then 'risky_first_name' else '' end||','
                        ||case when a.is_fraud_device_id = 1 
                                and a.bot_score_bucket > 60
                                and a.registration_hour in (0,2,4,5,6,6,8,11,12,14,14,14,15,15,17,19,19,20,21)
                            then 'fraud_dev_bot_reg_hr' else '' end||','
                        ||case when a.count_accounts >= 15 then 'device_linked_ge_15_accts' else '' end||','
                    ) as rules_triggered
                    ,a.*
                from (
                    select
                        --layer 0 risk rule scores based on sql
                        --high risk rules
                        case when m.prev_blocked_email_handle = 1 then 200 else 0 end
                            + case when m.uncommon_domain = 1 then 10 else 0 end
                            + case when m.disposable_domain = 1 then 100 else 0 end
                            + case when m.bot_score_bucket > 90 then -25 else 0 end
                            + case when m.is_emulator = 1 then 25 else 0 end
                            + case when lower(m.origin) like 'web%'
                                or lower(m.origin) = 'ipod'
                                or lower(m.origin) = 'ipad'
                                or lower(m.origin) = 'unknown' 
                                    then 50 else 0 end
                            + case when m.likely_invalid_email = 1 then 100 else 0 end
                            + case when m.zip in ('32578','8260','34292','28097','32202','67108','37938','46617'
                                                ,'46601','85034','26181','30303','32327','45458','95119','39203')
                                then 100 else 0 end
                            + case when m.signup_method = 'email' then 10 else 0 end
                            + case when m.registration_hour in (0,17,18,19,20,21,22,23) then 10 else 0 end
                            + case when m.email_nums in ('63','82','38','55','43','44','97','56','36','93','85','59','31',
                                '29','99','754','40','21','74','33','60','27','65','45','95','50','79','90','76','78','32',
                                '19','68','87','2','26','42','49','64','456','73','71','732','67','88','61','72','643','89',
                                '76,56','46','15','96','54','34','66','83','48','18','84','35','91','12','69','98','8')
                                then 100 else 0 end
                            + case when m.count_email_nums > 5 then 200 else 0 end
                            + case when m.state_name in ('VA','NJ','OH','ME','NV','WI','IN','MA','WY','SC','DE','UT','CO',
                                                        'AR','MO','ND','KY','IA','HI','MD','NH','SD','AL','NE','RI','NM')
                                then -50 else 0 end
                            + case when m.emails_are_bouncing = 1 then 200 else 0 end
                            + case when m.is_fraud_device_id = 1 then 200 else 0 end
                            + case when m.is_fraud_ip_address = 1 then 200 else 0 end
                            + case when m.first_name in ('Author Pinor','Lucas Betty','Diane','Williams Sandy','Clinton','Williams'
                                                        ,'Clinton Dean','Author','Woolley','Josephine','Marisa','Paula Ross','debbie')
                                then 50 else 0 end

                            --rule combos (often rules for good users)
                            + case when m.bot_score_bucket > 60 
                                    and m.registration_hour in (0,2,4,5,6,6,8,11,12,14,14,14,15,15,17,19,19,20,21)
                                then -200 else 0 end
                            + case when m.state_name in ('MS','NE','IN','SC','AL','MS','DE','OK','AR','MN','NE','NE','NH','AL','NH','KS','AL','MN') 
                                    and m.registration_hour in (4,16,10,7,6,14,12,9,5,6,4,23,12,9,13,4,11,10)
                                then -200 else 0 end
                            + case when lower(m.origin) = 'iphone' and m.is_fraud_ip_address = 0 then -50 else 0 end
                            + case when m.is_fraud_device_id = 1 
                                    and m.bot_score_bucket > 60
                                    and m.registration_hour in (0,2,4,5,6,6,8,11,12,14,14,14,15,15,17,19,19,20,21)
                                then 100 else 0 end
                            + case when m.count_accounts >= 15 then 200 else 0 end
                        as score
                        ,m.*
                    from (
                        select
                            SPLIT(z.user_agent, '(')[OFFSET(0)] ua_part1
                            ,case when z.first_name like '% %' then 1 else 0 end as is_two_names
                            ,case when z.sb_email_handle = z.email_handle then 1 else 0 end as prev_blocked_email_handle
                            ,case when z.email_domain not in ('gmail.com','yahoo.com','aol.com','hotmail.com','icloud.com'
                                ,'outlook.com','comcast.net','mail.com','privaterelay.appleid.com'
                                ,'live.com','msn.com','att.net','sbcglobal.net','verizon.net','me.com'
                                ,'ymail.com','mailinator.com','gamil.com','protonmail.com','yandex.com'
                                ,'gmx.com','bellsouth.net')
                                    then 1 else 0 end as uncommon_domain
                            ,case when z.email_handle like '%`%' or z.email_handle like "%'%" or z.email_handle like '%,%' or z.email_handle like '%"%' 
                                    or z.email_handle like '%:%' or z.email_handle like '%;%' or z.email_handle like '%!%' or z.email_handle like '%+%' 
                                    or z.email_handle like '%=%' or z.email_handle like '%/%' or z.email_handle like '%(%' or z.email_handle like '%*%' 
                                    or z.email_handle like '%)%' or z.email_handle like '%<%' or z.email_handle like '%>%' or z.email_handle like '%]%' 
                                    or z.email_handle like '%[%' or z.email_handle like '%}%' or z.email_handle like '%{%' or z.email_handle like '%#%' 
                                    or z.email_handle like '%$%' or z.email_handle like '%^%' or z.email_handle like '%&%' 
                                    or z.email not like '%@%.%' or z.email like '%..%' or z.email like '%@%@%' or z.email like '%.@%' or z.email like '%@.%'
                                    or z.email_domain like '%.cm' or z.email_domain like '%.co' 
                                then 1 else 0 end as likely_invalid_email
                            ,case when z.email_domain in ('0815.ru','0wnd.net','0wnd.org','10minutemail.co.za','10minutemail.com','123-m.com','1fsdfdsfsdf.tk','1pad.de'
                                ,'20minutemail.com','21cn.com','2fdgdfgdfgdf.tk','2prong.com','30minutemail.com','33mail.com','3trtretgfrfe.tk','4gfdsgfdgfd.tk','4warding.com'
                                ,'5ghgfhfghfgh.tk','6hjgjhgkilkj.tk','6paq.com','7tags.com','9ox.net','a-bc.net','agedmail.com','ama-trade.de','amilegit.com','amiri.net'
                                ,'amiriindustries.com','anonmails.de','anonymbox.com','antichef.com','antichef.net','antireg.ru','antispam.de','antispammail.de','armyspy.com'
                                ,'artman-conception.com','azmeil.tk','baxomale.ht.cx','beefmilk.com','bigstring.com','binkmail.com','bio-muesli.net','bobmail.info','bodhi.lawlita.com'
                                ,'bofthew.com','bootybay.de','boun.cr','bouncr.com','breakthru.com','brefmail.com','bsnow.net','bspamfree.org','bugmenot.com','bund.us','burstmail.info'
                                ,'buymoreplays.com','byom.de','c2.hu','card.zp.ua','casualdx.com','cek.pm','centermail.com','centermail.net','chammy.info','childsavetrust.org'
                                ,'chogmail.com','choicemail1.com','clixser.com','cmail.net','cmail.org','coldemail.info','cool.fr.nf','courriel.fr.nf','courrieltemporaire.com'
                                ,'crapmail.org','cust.in','cuvox.de','d3p.dk','dacoolest.com','dandikmail.com','dayrep.com','dcemail.com','deadaddress.com','deadspam.com','delikkt.de'
                                ,'despam.it','despammed.com','devnullmail.com','dfgh.net','digitalsanctuary.com','dingbone.com','disposableaddress.com','disposableemailaddresses.com'
                                ,'disposableinbox.com','dispose.it','dispostable.com','dodgeit.com','dodgit.com','donemail.ru','dontreg.com','dontsendmespam.de','drdrb.net'
                                ,'dump-email.info','dumpandjunk.com','dumpyemail.com','e-mail.com','e-mail.org','e4ward.com','easytrashmail.com','einmalmail.de','einrot.com'
                                ,'eintagsmail.de','emailgo.de','emailias.com','emaillime.com','emailsensei.com','emailtemporanea.com','emailtemporanea.net','emailtemporar.ro'
                                ,'emailtemporario.com.br','emailthe.net','emailtmp.com','emailwarden.com','emailx.at.hm','emailxfer.com','emeil.in','emeil.ir','emz.net','ero-tube.org'
                                ,'evopo.com','explodemail.com','express.net.ua','eyepaste.com','fakeinbox.com','fakeinformation.com','fansworldwide.de','fantasymail.de','zoemail.net'
                                ,'fightallspam.com','filzmail.com','fivemail.de','fleckens.hu','frapmail.com','friendlymail.co.uk','fuckingduh.com','fudgerub.com','fyii.de','zomg.info'
                                ,'garliclife.com','gehensiemirnichtaufdensack.de','get2mail.fr','getairmail.com','getmails.eu','getonemail.com','giantmail.de','girlsundertheinfluence.com'
                                ,'gishpuppy.com','gmial.com','goemailgo.com','gotmail.net','gotmail.org','gotti.otherinbox.com','great-host.in','greensloth.com','grr.la','gsrv.co.uk'
                                ,'guerillamail.biz','guerillamail.com','guerrillamail.biz','guerrillamail.com','guerrillamail.de','guerrillamail.info','guerrillamail.net','guerrillamail.org'
                                ,'guerrillamailblock.com','gustr.com','harakirimail.com','hat-geld.de','hatespam.org','herp.in','hidemail.de','hidzz.com','hmamail.com','hopemail.biz'
                                ,'ieh-mail.de','ikbenspamvrij.nl','imails.info','inbax.tk','inbox.si','inboxalias.com','inboxclean.com','inboxclean.org','infocom.zp.ua','instant-mail.de'
                                ,'ip6.li','irish2me.com','iwi.net','jetable.com','jetable.fr.nf','jetable.net','jetable.org','jnxjn.com','jourrapide.com','jsrsolutions.com','kasmail.com'
                                ,'kaspop.com','killmail.com','killmail.net','klassmaster.com','klzlk.com','koszmail.pl','kurzepost.de','lawlita.com','letthemeatspam.com','lhsdv.com'
                                ,'lifebyfood.com','link2mail.net','litedrop.com','lol.ovpn.to','lolfreak.net','lookugly.com','lortemail.dk','lr78.com','lroid.com','lukop.dk','m21.cc'
                                ,'mail-filter.com','mail-temporaire.fr','mail.by','mail.mezimages.net','mail.zp.ua','mail1a.de','mail21.cc','mail2rss.org','mail333.com','mailbidon.com'
                                ,'mailbiz.biz','mailblocks.com','mailbucket.org','mailcat.biz','mailcatch.com','mailde.de','mailde.info','maildrop.cc','maileimer.de','mailexpire.com'
                                ,'mailfa.tk','mailforspam.com','mailfreeonline.com','mailguard.me','mailin8r.com','mailinater.com','mailinator.com','mailinator.net','mailinator.org'
                                ,'mailinator2.com','mailincubator.com','mailismagic.com','mailme.lv','mailme24.com','mailmetrash.com','mailmoat.com','mailms.com','mailnesia.com'
                                ,'mailnull.com','mailorg.org','mailpick.biz','mailrock.biz','mailscrap.com','mailshell.com','mailsiphon.com','mailtemp.info','mailtome.de','mailtothis.com'
                                ,'mailtrash.net','mailtv.net','mailtv.tv','mailzilla.com','makemetheking.com','manybrain.com','mbx.cc','mega.zik.dj','meinspamschutz.de','meltmail.com'
                                ,'messagebeamer.de','mezimages.net','ministry-of-silly-walks.de','mintemail.com','misterpinball.de','moncourrier.fr.nf','monemail.fr.nf','monmail.fr.nf'
                                ,'monumentmail.com','mt2009.com','mt2014.com','mycard.net.ua','mycleaninbox.net','mymail-in.net','mypacks.net','mypartyclip.de','myphantomemail.com'
                                ,'mysamp.de','mytempemail.com','mytempmail.com','mytrashmail.com','nabuma.com','neomailbox.com','nepwk.com','nervmich.net','nervtmich.net','netmails.com'
                                ,'netmails.net','neverbox.com','nice-4u.com','nincsmail.hu','nnh.com','no-spam.ws','noblepioneer.com','nomail.pw','nomail.xl.cx','nomail2me.com'
                                ,'nomorespamemails.com','nospam.ze.tc','nospam4.us','nospamfor.us','nospammail.net','notmailinator.com','nowhere.org','nowmymail.com','nurfuerspam.de'
                                ,'nus.edu.sg','objectmail.com','obobbo.com','odnorazovoe.ru','oneoffemail.com','onewaymail.com','onlatedotcom.info','online.ms','opayq.com','ordinaryamerican.net'
                                ,'otherinbox.com','ovpn.to','owlpic.com','pancakemail.com','pcusers.otherinbox.com','pjjkp.com','plexolan.de','poczta.onet.pl','politikerclub.de','poofy.org'
                                ,'pookmail.com','privacy.net','privatdemail.net','proxymail.eu','prtnx.com','putthisinyourspamdatabase.com','putthisinyourspamdatabase.com','qq.com'
                                ,'quickinbox.com','rcpt.at','reallymymail.com','realtyalerts.ca','recode.me','recursor.net','reliable-mail.com','rhyta.com','rmqkr.net','royal.net'
                                ,'rtrtr.com','s0ny.net','safe-mail.net','safersignup.de','safetymail.info','safetypost.de','saynotospams.com','schafmail.de','schrott-email.de','secretemail.de'
                                ,'secure-mail.biz','senseless-entertainment.com','services391.com','sharklasers.com','shieldemail.com','shiftmail.com','shitmail.me','shitware.nl','shmeriously.com'
                                ,'shortmail.net','sibmail.com','sinnlos-mail.de','slapsfromlastnight.com','slaskpost.se','smashmail.de','smellfear.com','snakemail.com','sneakemail.com'
                                ,'sneakmail.de','snkmail.com','sofimail.com','solvemail.info','sogetthis.com','soodonims.com','spam4.me','spamail.de','spamarrest.com','spambob.net','spambog.ru'
                                ,'spambox.us','spamcannon.com','spamcannon.net','spamcon.org','spamcorptastic.com','spamcowboy.com','spamcowboy.net','spamcowboy.org','spamday.com','spamex.com'
                                ,'spamfree.eu','spamfree24.com','spamfree24.de','spamfree24.org','spamgoes.in','spamgourmet.com','spamgourmet.net','spamgourmet.org','spamherelots.com'
                                ,'spamherelots.com','spamhereplease.com','spamhereplease.com','spamhole.com','spamify.com','spaml.de','spammotel.com','spamobox.com','spamslicer.com'
                                ,'spamspot.com','spamthis.co.uk','spamtroll.net','speed.1s.fr','spoofmail.de','stuffmail.de','super-auswahl.de','supergreatmail.com','supermailer.jp'
                                ,'superrito.com','superstachel.de','suremail.info','talkinator.com','teewars.org','teleworm.com','teleworm.us','temp-mail.org','temp-mail.ru','tempe-mail.com'
                                ,'tempemail.co.za','tempemail.com','tempemail.net','tempemail.net','tempinbox.co.uk','tempinbox.com','tempmail.eu','tempmaildemo.com','tempmailer.com'
                                ,'tempmailer.de','tempomail.fr','temporaryemail.net','temporaryforwarding.com','temporaryinbox.com','temporarymailaddress.com','tempthe.net','thankyou2010.com'
                                ,'thc.st','thelimestones.com','thisisnotmyrealemail.com','thismail.net','throwawayemailaddress.com','tilien.com','tittbit.in','tizi.com','tmailinator.com'
                                ,'toomail.biz','topranklist.de','tradermail.info','trash-mail.at','trash-mail.com','trash-mail.de','trash2009.com','trashdevil.com','trashemail.de'
                                ,'trashmail.at','trashmail.com','trashmail.de','trashmail.me','trashmail.net','trashmail.org','trashymail.com','trialmail.de','trillianpro.com','twinmail.de'
                                ,'tyldd.com','uggsrock.com','umail.net','uroid.com','us.af','venompen.com','veryrealemail.com','viditag.com','viralplays.com','vpn.st','vsimcard.com'
                                ,'vubby.com','wasteland.rfc822.org','webemail.me','weg-werf-email.de','wegwerf-emails.de','wegwerfadresse.de','wegwerfemail.com','wegwerfemail.de'
                                ,'wegwerfmail.de','wegwerfmail.info','wegwerfmail.net','wegwerfmail.org','wh4f.org','whyspam.me','willhackforfood.biz','willselfdestruct.com','winemaven.info'
                                ,'wronghead.com','www.e4ward.com','www.mailinator.com','wwwnew.eu','x.ip6.li','xagloo.com','xemaps.com','xents.com','xmaily.com','xoxy.net','yep.it'
                                ,'yogamaven.com','yopmail.com','yopmail.fr','yopmail.net','yourdomain.com','yuurok.com','z1p.biz','za.com','zehnminuten.de','zehnminutenmail.de','zippymail.info')
                                    then 1 else 0 end as disposable_domain
                            ,case when z.firewall_bot_score between 0 and 5 then 5
                                when z.firewall_bot_score between 6 and 10 then 10
                                when z.firewall_bot_score between 11 and 15 then 15
                                when z.firewall_bot_score between 16 and 20 then 20
                                when z.firewall_bot_score between 21 and 25 then 25
                                when z.firewall_bot_score between 26 and 30 then 30
                                when z.firewall_bot_score between 31 and 35 then 35
                                when z.firewall_bot_score between 36 and 40 then 40
                                when z.firewall_bot_score between 41 and 45 then 45
                                when z.firewall_bot_score between 46 and 50 then 50
                                when z.firewall_bot_score between 51 and 55 then 55
                                when z.firewall_bot_score between 56 and 60 then 60
                                when z.firewall_bot_score between 61 and 65 then 65
                                when z.firewall_bot_score between 66 and 70 then 70
                                when z.firewall_bot_score between 71 and 75 then 75
                                when z.firewall_bot_score between 76 and 80 then 80
                                when z.firewall_bot_score between 81 and 85 then 85
                                when z.firewall_bot_score between 86 and 90 then 90
                                when z.firewall_bot_score between 91 and 95 then 95
                                    else 100 end as bot_score_bucket
                            ,case when lower(z.user_agent) like '%bluestack%'
                                or lower(z.user_agent) like '%blue_stack%'
                                or lower(z.user_agent) like '%blue-stack%'
                                or lower(z.user_agent) like '%nox%'
                                or lower(z.user_agent) like '%memu%'
                                or lower(z.user_agent) like '%androidstudio%'
                                or lower(z.user_agent) like '%android_studio%'
                                or lower(z.user_agent) like '%android-studio%'
                                    then 1 else 0 end as is_emulator
                            ,case when z.length_email_handle between 0 and 10 then 10
                                when z.length_email_handle between 11 and 15 then 15
                                when z.length_email_handle between 16 and 20 then 20
                                when z.length_email_handle >= 21 then 21
                                    else 0 end as length_email_handle_bin
                            ,case when z.first_name like '%`%'
                                    or z.first_name like '%~%'
                                    or z.first_name like '%!%'
                                    or z.first_name like '%@%'
                                    or z.first_name like '%#%'
                                    or z.first_name like '%$%'
                                    or z.first_name like '%%%'
                                    or z.first_name like '%^%'
                                    or z.first_name like '%&%'
                                    or z.first_name like '%*%'
                                    or z.first_name like '%(%'
                                    or z.first_name like '%)%'
                                    or z.first_name like '%-%'
                                    or z.first_name like '%_%'
                                    or z.first_name like '%;%'
                                    or z.first_name like '%:%'
                                    or z.first_name like '%[%'
                                    or z.first_name like '%]%'
                                    or z.first_name like '%{%'
                                    or z.first_name like '%}%'
                                    or z.first_name like '%|%'
                                    or z.first_name like '%/%'
                                    or z.first_name like '%,%'
                                    or z.first_name like '%.%'
                                    or z.first_name like "%'%"
                                    or z.first_name like '%"%'
                                    or z.first_name like '%<%'
                                    or z.first_name like '%>%'
                                    or z.first_name like '%?%'
                                    or z.first_name like '%=%'
                                    or z.first_name like '%+%'
                                        then 1 else 0 end as first_name_spec_char
                            ,z.*
                        from (
                            select
                                distinct(u.header.user_id) as user_id
                                --,date(timestamp_trunc(l.first_login_date, day)) as first_login_date
                                ,date(timestamp_trunc(u.event_timestamp, day)) as registration_day
                                ,date(timestamp_trunc(u.event_timestamp, week)) as registration_week
                                ,cast(u.header.user_agent as string) as user_agent
                                ,zm.state_name
                                ,zm.zip
                                ,u.header.origin
                                ,u.header.ip_address as ip_address
                                ,u.header.firewall_bot_score as firewall_bot_score
                                ,u.header.device_id
                                --,u.header.device_fingerprint
                                ,u.first_name
                                ,u.signup_method
                                --,l.login_source
                                --,l.action
                                ,l.device_is_blacklisted
                                ,u.email_verified
                                ,u.username as email
                                --,u.event_timestamp as registration_time
                                --,l.first_login_date as first_login_time
                                --,date_diff(l.first_login_date, u.event_timestamp, millisecond) as reg_login_diff_ms
                                --,date_diff(l.first_login_date, u.event_timestamp, second) as reg_login_diff_sec
                                --,date_diff(l.first_login_date, u.event_timestamp, minute) as reg_login_diff_min
                                --,date_diff(l.first_login_date, u.event_timestamp, hour) as reg_login_diff_hour
                                --,date_diff(l.first_login_date, u.event_timestamp, day) as reg_login_diff_day
                                
                                --parse user agent colum
                                ,SPLIT(SPLIT(u.header.user_agent, 'OfferUp/')[SAFE_OFFSET(1)],' ')[SAFE_OFFSET(0)] as app_version
                                ,SPLIT(SPLIT(u.header.user_agent, 'build: ')[SAFE_OFFSET(1)],';')[SAFE_OFFSET(0)] as app_build
                                ,SPLIT(SPLIT(u.header.user_agent, 'OfferUp/')[SAFE_OFFSET(1)],'; ')[SAFE_OFFSET(1)] as device_type
                                ,REPLACE(SPLIT(SPLIT(SPLIT(u.header.user_agent, 'OfferUp/')[SAFE_OFFSET(1)],'; ')[SAFE_OFFSET(1)],'iPhone')[SAFE_OFFSET(1)],',','.') as iphone_type
                                ,SPLIT(SPLIT(u.header.user_agent, 'OfferUp/')[SAFE_OFFSET(1)],'; ')[SAFE_OFFSET(2)] as os
                                ,SPLIT(SPLIT(SPLIT(u.header.user_agent, 'OfferUp/')[SAFE_OFFSET(1)],'; ')[SAFE_OFFSET(2)],' ')[SAFE_OFFSET(0)] as os_type
                                ,SPLIT(SPLIT(SPLIT(u.header.user_agent, 'OfferUp/')[SAFE_OFFSET(1)],'; ')[SAFE_OFFSET(2)],' ')[SAFE_OFFSET(1)] as os_version
                                ,SPLIT(SPLIT(SPLIT(u.header.user_agent, 'OfferUp/')[SAFE_OFFSET(1)],'; ')[SAFE_OFFSET(3)],')')[SAFE_OFFSET(0)] as lang
                                --,case when l.ou_captcha_session_id is null then 0 else 1 end as captcha_exists
                                --,case when ver_email.email_verified_time < l.first_login_date then 1 else 0 end as email_verified
                                --,case when ver_phone.phone_verified_time < l.first_login_date then 1 else 0 end as phone_verified
                                --,case when ver_truyou.truyou_verified_time < l.first_login_date then 1 else 0 end as truyou_verified
    
                                ,cast(u.event_timestamp as date) as registration_date
                                ,EXTRACT(HOUR FROM TIMESTAMP(u.event_timestamp)) as registration_hour
                                ,length(REGEXP_EXTRACT(u.username, r"^[a-zA-Z0-9_.+-]+")) as length_email_handle
                                ,REGEXP_EXTRACT(u.username, r"^[a-zA-Z0-9_.+-]+") as email_handle
                                --,REGEXP_EXTRACT(u.username, r"@(.+)") as email_domain
                                ,d.email_domain
                                ,REGEXP_EXTRACT(u.username, r"(\.[^.]*)$") as top_level_domain
                                ,TRIM(REGEXP_REPLACE(u.email, r'[^\d]+', ','),",") as email_nums
                                ,case when cast(ROUND(CHAR_LENGTH(TRIM(REGEXP_REPLACE(u.email, r'[^\d]+', ','),",")),0) as int) >= 0
                                    then cast(ROUND(CHAR_LENGTH(TRIM(REGEXP_REPLACE(u.email, r'[^\d]+', ','),",")),0) as int) else 0 end as count_email_nums
                                ,case when lower(u.header.origin) in ('ipod','ipad','web_desktop') then 1 else 0 end as risky_origin
                                ,s.sb_email_handle
                                ,case when pp.picture_user_id is not null then 1 else 0 end as added_profile_pic
                                
                                --case statements and lookups
                                ,du.count_accounts
                                ,case when d.domain_user_id is not null then 1 else 0 end as disposable_domain_test
                                ,case when ip.fraud_ip_address is not null then 1 else 0 end as is_fraud_ip_address
                                ,case when did.fraud_device_id is not null then 1 else 0 end as is_fraud_device_id
                                ,case when f.flagged_user_id is not null then 1 else 0 end as emails_are_bouncing
                                ,case when s.softblocked_user_id is not null then 'softblocked' else 'not_softblocked' end as is_softblocked
                                ,case when s.softblocked_user_id is not null then 1 else 0 end as is_softblocked_tf
                            from `some_table` u
                            join (
                                select 
                                    distinct(safe_cast(user_id as int64)) as user_id
                                from `some_table_1`
                                where
                                    metric_name = 'Active User'
                                    and activity_date_pacific_time >= date_add(current_date, interval -365 day)
                                ) sleepers on u.header.user_id = sleepers.user_id
                            left join `some_table_2` as uzip on u.header.user_id = cast(uzip.user_id as int)
                            left join `some_table_3` as zm on zm.zip = uzip.geoid10
                            left join (
                                select
                                    u.header.device_id
                                    ,count(distinct(u.header.user_id)) as count_accounts
                                from `some_table` u
                                where
                                    u.event_timestamp is not null
                                group by
                                    1
                                ) du on u.header.device_id = du.device_id
                            left join (
                                select
                                    u.header.user_id as domain_user_id
                                    ,REGEXP_EXTRACT(u.username, r"@(.+)") as email_domain
                                from `some_table` u
                                join `some_table_3` d on d.domain = REGEXP_EXTRACT(u.username, r"@(.+)")
                                where
                                    u.event_timestamp is not null
                                ) d on u.header.user_id = d.domain_user_id
                            left join (
                                select
                                    pp.header.user_id as picture_user_id
                                    ,pp.event_timestamp
                                from `some_table_x` pp
                                where
                                    pp.event_timestamp is not null
                                ) pp on u.header.user_id = pp.picture_user_id
                                    and u.event_timestamp = pp.event_timestamp
                            left join ( --get login table data
                                select
                                    distinct(z.login_user_id) as login_user_id
                                    ,z.login_source
                                    ,z.action
                                    ,z.device_is_blacklisted
                                    ,z.ou_captcha_session_id
                                    ,z.first_login_date
                                    ,z.ip_address
                                    ,z.event_rank
                                from (
                                    select
                                        l.header.user_id as login_user_id
                                        ,l.login_source
                                        ,l.action
                                        ,l.device_is_blacklisted
                                        ,l.ou_captcha_session_id
                                        ,l.header.ip_address as ip_address
                                        ,row_number() over (partition by l.header.user_id order by l.event_timestamp desc) as event_rank
                                        ,min(l.event_timestamp) as first_login_date
                                    from `some_table_x` l
                                    where
                                        l.event_timestamp is not null
                                        and l.action = 'login'
                                    group by
                                        1,2,3,4,5,6,l.event_timestamp
                                    ) z
                                where
                                    z.event_rank = 1
                                order by
                                    1 desc
                                ) l on u.header.user_id = l.login_user_id
                            left join ( -- fraud ip addresses
                                select
                                    distinct(u.header.ip_address) as fraud_ip_address
                                from `some_table_x` s
                                join `some_table` u on s.header.user_id = u.header.user_id
                                where
                                    s.event_timestamp is not null
                                    and u.event_timestamp is not null
                                    and s.user_softblock_change_type = 'softblocked'
                                ) ip on u.header.ip_address = ip.fraud_ip_address
                            left join ( -- fraud device ids
                                select
                                    distinct(u.header.device_id) as fraud_device_id
                                from `some_table_x` s
                                join `some_table` u on s.header.user_id = u.header.user_id
                                where
                                    s.event_timestamp is not null
                                    and u.event_timestamp is not null
                                    and s.user_softblock_change_type = 'softblocked'
                                ) did on u.header.device_id = did.fraud_device_id
                            left join ( --get emails are bouncing column
                                select
                                    f.header.user_id as flagged_user_id
                                    ,max(f.flag_type) as flag_type
                                from `some_table_x` f
                                where
                                    f.event_timestamp is not null
                                    and f.flag_type in ('emails_are_bouncing','bouncing_email')
                                group by
                                    1
                                ) f on u.header.user_id = f.flagged_user_id
                            left join ( --get softblocks; definition: select distinct user_ids who were either softblocked or failed phone verification
                                --join only users who's latest event is softblocked
                                select
                                    distinct(z.softblocked_user_id) as softblocked_user_id
                                    ,z.sb_email_handle
                                    ,z.max_ts
                                from (
                                    select
                                        s.header.user_id as softblocked_user_id
                                        ,REGEXP_EXTRACT(u.email, r"^[a-zA-Z0-9_.+-]+") as sb_email_handle
                                        ,max(s.event_timestamp) as max_ts
                                    from `some_table_x` s
                                    join `some_table` u on s.header.user_id = u.header.user_id
                                    where
                                        s.event_timestamp is not null
                                        and u.event_timestamp is not null
                                        and s.user_softblock_change_type = 'softblocked'
                                    group by
                                        1,2
                                    ) z

                                union all 

                                --join users who did not pass phone verification
                                select 
                                    distinct(z.flagged_user_id) as softblocked_user_id
                                    ,z.sb_email_handle
                                    ,z.max_ts
                                from (
                                    select
                                        f.header.user_id as flagged_user_id
                                        ,REGEXP_EXTRACT(u.email, r"^[a-zA-Z0-9_.+-]+") as sb_email_handle
                                        ,max(f.event_timestamp) as max_ts
                                    from `some_table_x` f
                                    join `some_table` u on f.header.user_id = u.header.user_id
                                    where
                                        f.event_timestamp is not null
                                        and u.event_timestamp is not null
                                        and f.flag_type = 'restrict_all_access_phone_verify'
                                        and lower(f.change_type) = 'added'
                                    group by
                                        1,2
                                    ) z
                                ) s on s.softblocked_user_id = u.header.user_id
                            where
                                u.event_timestamp is not null
                                --and u.event_timestamp between '2021-08-10' and '2021-08-17'
                                and u.event_timestamp >= timestamp_add(CURRENT_TIMESTAMP(), interval -numdays day)
                                --and u.event_timestamp >= timestamp_add(CURRENT_TIMESTAMP(), interval -numhours hour)
                            ) z
                        ) m
                    ) a
                ) b
            ) c
        ) d
)

select
    case when d.risk_rules_triggered like '%rule_prev_blocked_email_handle%' then 1 else 0 end as rule_prev_blocked_email_handle
    ,case when d.risk_rules_triggered like '%rule_uncommon_domain%' then 1 else 0 end as rule_uncommon_domain
    ,case when d.risk_rules_triggered like '%rule_disposable_domain%' then 1 else 0 end as rule_disposable_domain
    ,case when d.risk_rules_triggered like '%rule_bot_score_bucket%' then 1 else 0 end as rule_bot_score_bucket
    ,case when d.risk_rules_triggered like '%rule_is_emulator%' then 1 else 0 end as rule_is_emulator
    ,case when d.risk_rules_triggered like '%rule_origin%' then 1 else 0 end as rule_origin
    ,case when d.risk_rules_triggered like '%rule_likely_invalid_email%' then 1 else 0 end as rule_likely_invalid_email
    ,case when d.risk_rules_triggered like '%rule_zip%' then 1 else 0 end as rule_zip
    ,case when d.risk_rules_triggered like '%rule_signup_method%' then 1 else 0 end as rule_signup_method
    ,case when d.risk_rules_triggered like '%rule_registration_hour%' then 1 else 0 end as rule_registration_hour
    ,case when d.risk_rules_triggered like '%rule_email_nums%' then 1 else 0 end as rule_email_nums
    ,case when d.risk_rules_triggered like '%rule_count_email_nums%' then 1 else 0 end as rule_count_email_nums
    ,case when d.risk_rules_triggered like '%rule_state_name%' then 1 else 0 end as rule_state_name
    ,case when d.risk_rules_triggered like '%rule_vemails_are_bouncing%' then 1 else 0 end as rule_emails_are_bouncing
    ,case when d.risk_rules_triggered like '%rule_is_fraud_device_id%' then 1 else 0 end as rule_is_fraud_device_id
    ,case when d.risk_rules_triggered like '%rule_is_fraud_ip_address%' then 1 else 0 end as rule_is_fraud_ip_address
    ,case when d.risk_rules_triggered like '%rule_fraud_ip_reg_hour%' then 1 else 0 end as rule_fraud_ip_reg_hour
    ,case when d.risk_rules_triggered like '%rule_state_name_reg_hour%' then 1 else 0 end as rule_state_name_reg_hour
    ,case when d.risk_rules_triggered like '%rule_origin_fraud_ip%' then 1 else 0 end as rule_origin_fraud_ip
    ,case when d.risk_rules_triggered like '%rule_low_risk_rules_triggered%' then 1 else 0 end as rule_low_risk_rules_triggered
    ,case when d.risk_rules_triggered like '%rule_risky_first_name%' then 1 else 0 end as rule_risky_first_name
    ,case when d.risk_rules_triggered like '%fraud_dev_bot_reg_hr%' then 1 else 0 end as rule_fraud_dev_bot_reg_hr
    ,case when d.risk_rules_triggered like '%device_linked_ge_15_accts%' then 1 else 0 end as rule_device_linked_ge_15_accts
    ,d.*
from data d
"""

#to dataframe
df = gcp_client.query(query).to_dataframe()

#display original df
# print('Original Sql df:')
# display(df.head(2))


end_time1 = datetime.now()
print('Query completed and inserted into dataframe. ','Duration: {}'.format(end_time1 - start_time))

#try dask
# df = dd(gcp_client.query(query))

#random filters
# df = df.fillna(value=np.nan)
# df = df.fillna('0')
# df['ip_address'] = df['ip_address'].replace('0', '0.0.0.0')
df['ip_address'] = df['ip_address'].replace(' ', '0.0.0.0')
df['ip_address'] = df['ip_address'].replace('nan', '0.0.0.0')
df['ip_address'] = df['ip_address'].replace('NaN', '0.0.0.0')
df['ip_address'] = df['ip_address'].fillna('0.0.0.0')
df['length_email_handle'] = df['length_email_handle'].fillna(0)
# df['top_level_domain'] = df['top_level_domain'].fillna(0)
df['email'] = df['email'].astype('str')
df['risk_scores'] = df['risk_scores'].astype('str')

#apply classify function for gibberish score  
df['classify_email_score'] = round(df.apply(lambda row: classify(row['email']), axis = 1),0)
df['classify_handle_score'] = round(df.apply(lambda row: classify(row['email_handle']), axis = 1),0)
df['classify_domain_score'] = round(df.apply(lambda row: classify(row['email_domain']), axis = 1),0)

#convert cols to strings that need to be
# df['classify_email_score'] = df['classify_email_score'].astype('str')
# df['classify_handle_score'] = df['classify_handle_score'].astype('str')
# df['classify_domain_score'] = df['classify_domain_score'].astype('str')

#apply strangeness function for gibberish score  
# df['strangeness_email_score'] = round(df.apply(lambda row: strangeness(row['email']), axis = 1),1)
# df['strangeness_handle_score'] = round(df.apply(lambda row: strangeness(row['email_handle']), axis = 1),1)
# df['strangeness_domain_score'] = round(df.apply(lambda row: strangeness(row['email_domain']), axis = 1),1)

end_time2 = datetime.now()
print('Gibberish columns created. ','Duration: {}'.format(end_time2 - start_time))


#Add new rules and update score based on rules
#layer 1 risk rule scores with python based rules
#apply score changes based on variables not found in the data (string gibberish for example):
m = df['classify_handle_score'].between(92, 99)
df.loc[m, 'risk_score'] = df.loc[m, 'risk_score'] + 75
m = ((df['rule_signup_method'] == 1)
     & (df['rule_state_name'] == 1)
     & (df['rule_is_fraud_device_id'] == 1)
     & (df['rule_fraud_ip_reg_hour'] == 1))
df.loc[m, 'risk_score'] = df.loc[m, 'risk_score'] + 100

#append python based rules to rules triggered column
m = df['classify_handle_score'].between(92, 99)
df.loc[m, 'risk_rules_triggered'] = df.loc[m, 'risk_rules_triggered'] + ', classify_handle_score'
m = ((df['rule_signup_method'] == 1)
     & (df['rule_state_name'] == 1)
     & (df['rule_is_fraud_device_id'] == 1)
     & (df['rule_fraud_ip_reg_hour'] == 1))
df.loc[m, 'risk_rules_triggered'] = df.loc[m, 'risk_rules_triggered'] + ', rule_signup_state_dev_ip_hour'

#create aggregate columns for python based rules
df["classify_handle_score_rule"] = df["risk_rules_triggered"].map(lambda x: 1 if "classify_handle_score" in str(x) else 0)
df["rule_signup_state_dev_ip_hour"] = df["risk_rules_triggered"].map(lambda x: 1 if "rule_signup_state_dev_ip_hour" in str(x) else 0)


end_time3 = datetime.now()
print('New rules created, applied to score and appended to risk rules triggered col. ','Duration: {}'.format(end_time3 - start_time))


#*************Create Agg df here***************

#create agg df for rules metrics per day
agg_df_day = df.groupby('registration_day')[['classify_handle_score_rule','rule_prev_blocked_email_handle'
                                 ,'rule_uncommon_domain','rule_disposable_domain','rule_bot_score_bucket'
                                 ,'rule_is_emulator','rule_origin','rule_likely_invalid_email','rule_zip'
                                 ,'rule_signup_method','rule_registration_hour','rule_email_nums'
                                 ,'rule_count_email_nums','rule_state_name','rule_emails_are_bouncing'
                                 ,'rule_is_fraud_device_id','rule_is_fraud_ip_address','rule_fraud_ip_reg_hour'
                                 ,'rule_state_name_reg_hour','rule_origin_fraud_ip','rule_low_risk_rules_triggered'
                                 ,'rule_risky_first_name','rule_signup_state_dev_ip_hour','rule_fraud_dev_bot_reg_hr'
                                 ,'rule_device_linked_ge_15_accts']].agg('sum').reset_index()
agg_df_day = pd.melt(agg_df_day
                 ,id_vars =['registration_day']
                 ,value_vars =['classify_handle_score_rule',
                               'rule_prev_blocked_email_handle', 'rule_uncommon_domain', 'rule_disposable_domain',
                               'rule_bot_score_bucket', 'rule_is_emulator', 'rule_origin', 'rule_likely_invalid_email',
                               'rule_zip', 'rule_signup_method', 'rule_registration_hour', 'rule_email_nums',
                               'rule_count_email_nums', 'rule_state_name', 'rule_emails_are_bouncing',
                               'rule_is_fraud_device_id', 'rule_is_fraud_ip_address', 'rule_fraud_ip_reg_hour',
                               'rule_state_name_reg_hour', 'rule_origin_fraud_ip', 'rule_low_risk_rules_triggered',
                               'rule_risky_first_name','rule_signup_state_dev_ip_hour','rule_fraud_dev_bot_reg_hr',
                               'rule_device_linked_ge_15_accts'])
agg_df_day.columns = ['registration_day','rule','metric_value']
agg_df_day['weekly_vol_pct_change'] = agg_df_day['metric_value'].pct_change()
agg_df_day['prev_weekly_vol_pct_change'] = agg_df_day.groupby(['rule'])['weekly_vol_pct_change'].transform('shift')

#create agg df for rules metrics per week
agg_df_week = df.groupby('registration_week')[['classify_handle_score_rule','rule_prev_blocked_email_handle'
                                 ,'rule_uncommon_domain','rule_disposable_domain','rule_bot_score_bucket'
                                 ,'rule_is_emulator','rule_origin','rule_likely_invalid_email','rule_zip'
                                 ,'rule_signup_method','rule_registration_hour','rule_email_nums'
                                 ,'rule_count_email_nums','rule_state_name','rule_emails_are_bouncing'
                                 ,'rule_is_fraud_device_id','rule_is_fraud_ip_address','rule_fraud_ip_reg_hour'
                                 ,'rule_state_name_reg_hour','rule_origin_fraud_ip','rule_low_risk_rules_triggered'
                                 ,'rule_risky_first_name','rule_signup_state_dev_ip_hour','rule_fraud_dev_bot_reg_hr'
                                 ,'rule_device_linked_ge_15_accts']].agg('sum').reset_index()
agg_df_week = pd.melt(agg_df_week
                 ,id_vars =['registration_week']
                 ,value_vars =['classify_handle_score_rule',
                               'rule_prev_blocked_email_handle', 'rule_uncommon_domain', 'rule_disposable_domain',
                               'rule_bot_score_bucket', 'rule_is_emulator', 'rule_origin', 'rule_likely_invalid_email',
                               'rule_zip', 'rule_signup_method', 'rule_registration_hour', 'rule_email_nums',
                               'rule_count_email_nums', 'rule_state_name', 'rule_emails_are_bouncing',
                               'rule_is_fraud_device_id', 'rule_is_fraud_ip_address', 'rule_fraud_ip_reg_hour',
                               'rule_state_name_reg_hour', 'rule_origin_fraud_ip', 'rule_low_risk_rules_triggered',
                               'rule_risky_first_name','rule_signup_state_dev_ip_hour','rule_fraud_dev_bot_reg_hr',
                               'rule_device_linked_ge_15_accts'])
agg_df_week.columns = ['registration_week','rule','metric_value']
agg_df_week['weekly_vol_pct_change'] = agg_df_week['metric_value'].pct_change()
agg_df_week['prev_weekly_vol_pct_change'] = agg_df_week.groupby(['rule'])['weekly_vol_pct_change'].transform('shift')


end_time4 = datetime.now()
print('Agg df per day/week created. ','Duration: {}'.format(end_time4 - start_time))

#apply ip_checker
#too compute intesive - it works but it takes forever

#tried parallel distribution with dask, but don't know how to use it very well on gcp, or in general :)
# df['ip_checker'] = df.apply(lambda row: strangeness(row['email_domain']), axis = 1),2)

#create rule metrics df grouped by week
#create rule df grouped by week
rule_df = df[['registration_week','is_softblocked','is_softblocked_tf','classify_handle_score_rule','rule_prev_blocked_email_handle'
              ,'rule_uncommon_domain','rule_disposable_domain','rule_bot_score_bucket','rule_is_emulator','rule_origin','rule_likely_invalid_email','rule_zip'
              ,'rule_signup_method','rule_registration_hour','rule_email_nums','rule_count_email_nums','rule_state_name','rule_emails_are_bouncing'
              ,'rule_is_fraud_device_id','rule_is_fraud_ip_address','rule_fraud_ip_reg_hour','rule_state_name_reg_hour','rule_origin_fraud_ip'
              ,'rule_low_risk_rules_triggered','rule_risky_first_name','rule_signup_state_dev_ip_hour','rule_fraud_dev_bot_reg_hr'
              ,'rule_device_linked_ge_15_accts']]

#classify_handle_score_rule
g = rule_df.groupby('registration_week')
grouped_df = g.apply(lambda x: x[x['classify_handle_score_rule'] == 0]['classify_handle_score_rule'].count()).reset_index()
grouped_df.columns = ['registration_week','classify_handle_score_0']
classify_handle_score_1 = g.apply(lambda x: x[x['classify_handle_score_rule'] == 1]['classify_handle_score_rule'].count()).reset_index()
classify_handle_score_1.columns = ['registration_week','classify_handle_score_1']
classify_handle_score_1 = list(classify_handle_score_1['classify_handle_score_1'])
grouped_df['classify_handle_score_1'] = classify_handle_score_1
grouped_df['classify_handle_score_total'] = grouped_df['classify_handle_score_1'] + grouped_df['classify_handle_score_0']
grouped_df['classify_handle_score_sb_rate'] = grouped_df['classify_handle_score_1'] / grouped_df['classify_handle_score_total']
grouped_df['classify_handle_score_prev_wk_sb_rate'] = grouped_df['classify_handle_score_sb_rate'].shift()
grouped_df['classify_handle_score_sb_rate_pct_chg'] = grouped_df['classify_handle_score_prev_wk_sb_rate'].pct_change()

#prev_blocked_email_handle
grouped_df1 = g.apply(lambda x: x[x['rule_prev_blocked_email_handle'] == 0]['rule_prev_blocked_email_handle'].count()).reset_index()
grouped_df1.columns = ['registration_week','prev_blocked_email_handle_0']
prev_blocked_email_handle_1 = g.apply(lambda x: x[x['rule_prev_blocked_email_handle'] == 1]['rule_prev_blocked_email_handle'].count()).reset_index()
prev_blocked_email_handle_1.columns = ['registration_week','prev_blocked_email_handle_1']
prev_blocked_email_handle_1 = list(prev_blocked_email_handle_1['prev_blocked_email_handle_1'])
grouped_df1['prev_blocked_email_handle_1'] = prev_blocked_email_handle_1
grouped_df1['prev_blocked_email_handle_total'] = grouped_df1['prev_blocked_email_handle_1'] + grouped_df1['prev_blocked_email_handle_0']
grouped_df1['prev_blocked_email_handle_1_sb_rate'] = grouped_df1['prev_blocked_email_handle_1'] / grouped_df1['prev_blocked_email_handle_total']
grouped_df1['prev_blocked_email_handle_prev_wk_sb_rate'] = grouped_df1['prev_blocked_email_handle_1_sb_rate'].shift()
grouped_df1['prev_blocked_email_handle_sb_rate_pct_chg'] = grouped_df1['prev_blocked_email_handle_prev_wk_sb_rate'].pct_change()
grouped_df = pd.merge(grouped_df, grouped_df1, left_on='registration_week', right_on='registration_week')

#uncommon_domain
grouped_df2 = g.apply(lambda x: x[x['rule_uncommon_domain'] == 0]['rule_uncommon_domain'].count()).reset_index()
grouped_df2.columns = ['registration_week','uncommon_domain_0']
uncommon_domain_1 = g.apply(lambda x: x[x['rule_uncommon_domain'] == 1]['rule_uncommon_domain'].count()).reset_index()
uncommon_domain_1.columns = ['registration_week','uncommon_domain_1']
uncommon_domain_1 = list(uncommon_domain_1['uncommon_domain_1'])
grouped_df2['uncommon_domain_1'] = uncommon_domain_1
grouped_df2['uncommon_domain_total'] = grouped_df2['uncommon_domain_1'] + grouped_df2['uncommon_domain_0']
grouped_df2['uncommon_domain_1_sb_rate'] = grouped_df2['uncommon_domain_1'] / grouped_df2['uncommon_domain_total']
grouped_df2['uncommon_domain_prev_wk_sb_rate'] = grouped_df2['uncommon_domain_1_sb_rate'].shift()
grouped_df2['uncommon_domain_sb_rate_pct_chg'] = grouped_df2['uncommon_domain_prev_wk_sb_rate'].pct_change()
grouped_df = pd.merge(grouped_df, grouped_df2, left_on='registration_week', right_on='registration_week')

#disposable_domain
grouped_df3 = g.apply(lambda x: x[x['rule_disposable_domain'] == 0]['rule_disposable_domain'].count()).reset_index()
grouped_df3.columns = ['registration_week','disposable_domain_0']
disposable_domain_1 = g.apply(lambda x: x[x['rule_disposable_domain'] == 1]['rule_disposable_domain'].count()).reset_index()
disposable_domain_1.columns = ['registration_week','disposable_domain_1']
disposable_domain_1 = list(disposable_domain_1['disposable_domain_1'])
grouped_df3['disposable_domain_1'] = uncommon_domain_1
grouped_df3['disposable_domain_total'] = grouped_df3['disposable_domain_1'] + grouped_df3['disposable_domain_0']
grouped_df3['disposable_domain_1_sb_rate'] = grouped_df3['disposable_domain_1'] / grouped_df3['disposable_domain_total']
grouped_df3['disposable_domain_prev_wk_sb_rate'] = grouped_df3['disposable_domain_1_sb_rate'].shift()
grouped_df3['disposable_domain_sb_rate_pct_chg'] = grouped_df3['disposable_domain_prev_wk_sb_rate'].pct_change()
grouped_df = pd.merge(grouped_df, grouped_df3, left_on='registration_week', right_on='registration_week')

#bot_score_bucket
grouped_df4 = g.apply(lambda x: x[x['rule_bot_score_bucket'] == 0]['rule_bot_score_bucket'].count()).reset_index()
grouped_df4.columns = ['registration_week','bot_score_bucket_0']
bot_score_bucket_1 = g.apply(lambda x: x[x['rule_bot_score_bucket'] == 1]['rule_bot_score_bucket'].count()).reset_index()
bot_score_bucket_1.columns = ['registration_week','bot_score_bucket_1']
bot_score_bucket_1 = list(bot_score_bucket_1['bot_score_bucket_1'])
grouped_df4['bot_score_bucket_1'] = bot_score_bucket_1
grouped_df4['bot_score_bucket_total'] = grouped_df4['bot_score_bucket_1'] + grouped_df4['bot_score_bucket_0']
grouped_df4['bot_score_bucket_1_sb_rate'] = grouped_df4['bot_score_bucket_1'] / grouped_df4['bot_score_bucket_total']
grouped_df4['bot_score_bucket_prev_wk_sb_rate'] = grouped_df4['bot_score_bucket_1_sb_rate'].shift()
grouped_df4['bot_score_bucket_sb_rate_pct_chg'] = grouped_df4['bot_score_bucket_prev_wk_sb_rate'].pct_change()
grouped_df = pd.merge(grouped_df, grouped_df4, left_on='registration_week', right_on='registration_week')

#is_emulator
grouped_df5 = g.apply(lambda x: x[x['rule_is_emulator'] == 0]['rule_is_emulator'].count()).reset_index()
grouped_df5.columns = ['registration_week','is_emulator_0']
is_emulator_1 = g.apply(lambda x: x[x['rule_is_emulator'] == 1]['rule_is_emulator'].count()).reset_index()
is_emulator_1.columns = ['registration_week','is_emulator_1']
is_emulator_1 = list(is_emulator_1['is_emulator_1'])
grouped_df5['is_emulator_1'] = bot_score_bucket_1
grouped_df5['is_emulator_total'] = grouped_df5['is_emulator_1'] + grouped_df5['is_emulator_0']
grouped_df5['is_emulator_1_sb_rate'] = grouped_df5['is_emulator_1'] / grouped_df5['is_emulator_total']
grouped_df5['is_emulator_prev_wk_sb_rate'] = grouped_df5['is_emulator_1_sb_rate'].shift()
grouped_df5['is_emulator_sb_rate_pct_chg'] = grouped_df5['is_emulator_prev_wk_sb_rate'].pct_change()
grouped_df = pd.merge(grouped_df, grouped_df5, left_on='registration_week', right_on='registration_week')

#origin
grouped_df6 = g.apply(lambda x: x[x['rule_origin'] == 0]['rule_origin'].count()).reset_index()
grouped_df6.columns = ['registration_week','origin_0']
origin_1 = g.apply(lambda x: x[x['rule_origin'] == 1]['rule_origin'].count()).reset_index()
origin_1.columns = ['registration_week','origin_1']
origin_1 = list(origin_1['origin_1'])
grouped_df6['origin_1'] = origin_1
grouped_df6['origin_total'] = grouped_df6['origin_1'] + grouped_df6['origin_0']
grouped_df6['origin_1_sb_rate'] = grouped_df6['origin_1'] / grouped_df6['origin_total']
grouped_df6['origin_prev_wk_sb_rate'] = grouped_df6['origin_1_sb_rate'].shift()
grouped_df6['origin_sb_rate_pct_chg'] = grouped_df6['origin_prev_wk_sb_rate'].pct_change()
grouped_df = pd.merge(grouped_df, grouped_df6, left_on='registration_week', right_on='registration_week')

#likely_invalid_email
grouped_df7 = g.apply(lambda x: x[x['rule_likely_invalid_email'] == 0]['rule_likely_invalid_email'].count()).reset_index()
grouped_df7.columns = ['registration_week','likely_invalid_email_0']
likely_invalid_email_1 = g.apply(lambda x: x[x['rule_likely_invalid_email'] == 1]['rule_likely_invalid_email'].count()).reset_index()
likely_invalid_email_1.columns = ['registration_week','likely_invalid_email_1']
likely_invalid_email_1 = list(likely_invalid_email_1['likely_invalid_email_1'])
grouped_df7['likely_invalid_email_1'] = likely_invalid_email_1
grouped_df7['likely_invalid_email_total'] = grouped_df7['likely_invalid_email_1'] + grouped_df7['likely_invalid_email_0']
grouped_df7['likely_invalid_email_1_sb_rate'] = grouped_df7['likely_invalid_email_1'] / grouped_df7['likely_invalid_email_total']
grouped_df7['likely_invalid_email_prev_wk_sb_rate'] = grouped_df7['likely_invalid_email_1_sb_rate'].shift()
grouped_df7['likely_invalid_email_sb_rate_pct_chg'] = grouped_df7['likely_invalid_email_prev_wk_sb_rate'].pct_change()
grouped_df = pd.merge(grouped_df, grouped_df7, left_on='registration_week', right_on='registration_week')

#zip
grouped_df8 = g.apply(lambda x: x[x['rule_zip'] == 0]['rule_zip'].count()).reset_index()
grouped_df8.columns = ['registration_week','zip_0']
zip_1 = g.apply(lambda x: x[x['rule_zip'] == 1]['rule_zip'].count()).reset_index()
zip_1.columns = ['registration_week','zip_1']
zip_1 = list(zip_1['zip_1'])
grouped_df8['zip_1'] = zip_1
grouped_df8['zip_total'] = grouped_df8['zip_1'] + grouped_df8['zip_0']
grouped_df8['zip_1_sb_rate'] = grouped_df8['zip_1'] / grouped_df8['zip_total']
grouped_df8['zip_prev_wk_sb_rate'] = grouped_df8['zip_1_sb_rate'].shift()
grouped_df8['zip_sb_rate_pct_chg'] = grouped_df8['zip_prev_wk_sb_rate'].pct_change()
grouped_df = pd.merge(grouped_df, grouped_df8, left_on='registration_week', right_on='registration_week')

#signup_method
grouped_df9 = g.apply(lambda x: x[x['rule_signup_method'] == 0]['rule_signup_method'].count()).reset_index()
grouped_df9.columns = ['registration_week','signup_method_0']
signup_method_1 = g.apply(lambda x: x[x['rule_signup_method'] == 1]['rule_signup_method'].count()).reset_index()
signup_method_1.columns = ['registration_week','signup_method_1']
signup_method_1 = list(signup_method_1['signup_method_1'])
grouped_df9['signup_method_1'] = signup_method_1
grouped_df9['signup_method_total'] = grouped_df9['signup_method_1'] + grouped_df9['signup_method_0']
grouped_df9['signup_method_1_sb_rate'] = grouped_df9['signup_method_1'] / grouped_df9['signup_method_total']
grouped_df9['signup_method_prev_wk_sb_rate'] = grouped_df9['signup_method_1_sb_rate'].shift()
grouped_df9['signup_method_sb_rate_pct_chg'] = grouped_df9['signup_method_prev_wk_sb_rate'].pct_change()
grouped_df = pd.merge(grouped_df, grouped_df9, left_on='registration_week', right_on='registration_week')

#registration_hour
grouped_df10 = g.apply(lambda x: x[x['rule_registration_hour'] == 0]['rule_registration_hour'].count()).reset_index()
grouped_df10.columns = ['registration_week','registration_hour_0']
registration_hour_1 = g.apply(lambda x: x[x['rule_registration_hour'] == 1]['rule_registration_hour'].count()).reset_index()
registration_hour_1.columns = ['registration_week','registration_hour_1']
registration_hour_1 = list(registration_hour_1['registration_hour_1'])
grouped_df10['registration_hour_1'] = registration_hour_1
grouped_df10['registration_hour_total'] = grouped_df10['registration_hour_1'] + grouped_df10['registration_hour_0']
grouped_df10['registration_hour_1_sb_rate'] = grouped_df10['registration_hour_1'] / grouped_df10['registration_hour_total']
grouped_df10['registration_hour_prev_wk_sb_rate'] = grouped_df10['registration_hour_1_sb_rate'].shift()
grouped_df10['registration_hour_sb_rate_pct_chg'] = grouped_df10['registration_hour_prev_wk_sb_rate'].pct_change()
grouped_df = pd.merge(grouped_df, grouped_df10, left_on='registration_week', right_on='registration_week')

#email_nums
grouped_df11 = g.apply(lambda x: x[x['rule_email_nums'] == 0]['rule_email_nums'].count()).reset_index()
grouped_df11.columns = ['registration_week','email_nums_0']
email_nums_1 = g.apply(lambda x: x[x['rule_email_nums'] == 1]['rule_email_nums'].count()).reset_index()
email_nums_1.columns = ['registration_week','email_nums_1']
email_nums_1 = list(email_nums_1['email_nums_1'])
grouped_df11['email_nums_1'] = email_nums_1
grouped_df11['email_nums_total'] = grouped_df11['email_nums_1'] + grouped_df11['email_nums_0']
grouped_df11['email_nums_1_sb_rate'] = grouped_df11['email_nums_1'] / grouped_df11['email_nums_total']
grouped_df11['email_nums_prev_wk_sb_rate'] = grouped_df11['email_nums_1_sb_rate'].shift()
grouped_df11['email_nums_sb_rate_pct_chg'] = grouped_df11['email_nums_prev_wk_sb_rate'].pct_change()
grouped_df = pd.merge(grouped_df, grouped_df11, left_on='registration_week', right_on='registration_week')

#count_email_nums
grouped_df12 = g.apply(lambda x: x[x['rule_count_email_nums'] == 0]['rule_count_email_nums'].count()).reset_index()
grouped_df12.columns = ['registration_week','count_email_nums_0']
count_email_nums_1 = g.apply(lambda x: x[x['rule_count_email_nums'] == 1]['rule_count_email_nums'].count()).reset_index()
count_email_nums_1.columns = ['registration_week','count_email_nums_1']
count_email_nums_1 = list(count_email_nums_1['count_email_nums_1'])
grouped_df12['count_email_nums_1'] = count_email_nums_1
grouped_df12['count_email_nums_total'] = grouped_df12['count_email_nums_1'] + grouped_df12['count_email_nums_0']
grouped_df12['count_email_nums_1_sb_rate'] = grouped_df12['count_email_nums_1'] / grouped_df12['count_email_nums_total']
grouped_df12['count_email_nums_prev_wk_sb_rate'] = grouped_df12['count_email_nums_1_sb_rate'].shift()
grouped_df12['count_email_nums_sb_rate_pct_chg'] = grouped_df12['count_email_nums_prev_wk_sb_rate'].pct_change()
grouped_df = pd.merge(grouped_df, grouped_df12, left_on='registration_week', right_on='registration_week')

#state_name
grouped_df13 = g.apply(lambda x: x[x['rule_state_name'] == 0]['rule_state_name'].count()).reset_index()
grouped_df13.columns = ['registration_week','state_name_0']
state_name_1 = g.apply(lambda x: x[x['rule_state_name'] == 1]['rule_state_name'].count()).reset_index()
state_name_1.columns = ['registration_week','state_name_1']
state_name_1 = list(state_name_1['state_name_1'])
grouped_df13['state_name_1'] = state_name_1
grouped_df13['state_name_total'] = grouped_df13['state_name_1'] + grouped_df13['state_name_0']
grouped_df13['state_name_1_sb_rate'] = grouped_df13['state_name_1'] / grouped_df13['state_name_total']
grouped_df13['state_name_prev_wk_sb_rate'] = grouped_df13['state_name_1_sb_rate'].shift()
grouped_df13['state_name_sb_rate_pct_chg'] = grouped_df13['state_name_prev_wk_sb_rate'].pct_change()
grouped_df = pd.merge(grouped_df, grouped_df13, left_on='registration_week', right_on='registration_week')

#emails_are_bouncing
grouped_df14 = g.apply(lambda x: x[x['rule_emails_are_bouncing'] == 0]['rule_emails_are_bouncing'].count()).reset_index()
grouped_df14.columns = ['registration_week','emails_are_bouncing_0']
emails_are_bouncing_1 = g.apply(lambda x: x[x['rule_emails_are_bouncing'] == 1]['rule_emails_are_bouncing'].count()).reset_index()
emails_are_bouncing_1.columns = ['registration_week','emails_are_bouncing_1']
emails_are_bouncing_1 = list(emails_are_bouncing_1['emails_are_bouncing_1'])
grouped_df14['emails_are_bouncing_1'] = emails_are_bouncing_1
grouped_df14['emails_are_bouncing_total'] = grouped_df14['emails_are_bouncing_1'] + grouped_df14['emails_are_bouncing_0']
grouped_df14['emails_are_bouncing_1_sb_rate'] = grouped_df14['emails_are_bouncing_1'] / grouped_df14['emails_are_bouncing_total']
grouped_df14['emails_are_bouncing_prev_wk_sb_rate'] = grouped_df14['emails_are_bouncing_1_sb_rate'].shift()
grouped_df14['emails_are_bouncing_sb_rate_pct_chg'] = grouped_df14['emails_are_bouncing_prev_wk_sb_rate'].pct_change()
grouped_df = pd.merge(grouped_df, grouped_df14, left_on='registration_week', right_on='registration_week')

#is_fraud_device_id
grouped_df15 = g.apply(lambda x: x[x['rule_is_fraud_device_id'] == 0]['rule_is_fraud_device_id'].count()).reset_index()
grouped_df15.columns = ['registration_week','is_fraud_device_id_0']
is_fraud_device_id_1 = g.apply(lambda x: x[x['rule_is_fraud_device_id'] == 1]['rule_is_fraud_device_id'].count()).reset_index()
is_fraud_device_id_1.columns = ['registration_week','is_fraud_device_id_1']
is_fraud_device_id_1 = list(is_fraud_device_id_1['is_fraud_device_id_1'])
grouped_df15['is_fraud_device_id_1'] = is_fraud_device_id_1
grouped_df15['is_fraud_device_id_total'] = grouped_df15['is_fraud_device_id_1'] + grouped_df15['is_fraud_device_id_0']
grouped_df15['is_fraud_device_id_1_sb_rate'] = grouped_df15['is_fraud_device_id_1'] / grouped_df15['is_fraud_device_id_total']
grouped_df15['is_fraud_device_id_prev_wk_sb_rate'] = grouped_df15['is_fraud_device_id_1_sb_rate'].shift()
grouped_df15['is_fraud_device_id_sb_rate_pct_chg'] = grouped_df15['is_fraud_device_id_prev_wk_sb_rate'].pct_change()
grouped_df = pd.merge(grouped_df, grouped_df15, left_on='registration_week', right_on='registration_week')

#is_fraud_ip_address
grouped_df16 = g.apply(lambda x: x[x['rule_is_fraud_ip_address'] == 0]['rule_is_fraud_ip_address'].count()).reset_index()
grouped_df16.columns = ['registration_week','is_fraud_ip_address_0']
is_fraud_ip_address_1 = g.apply(lambda x: x[x['rule_is_fraud_ip_address'] == 1]['rule_is_fraud_ip_address'].count()).reset_index()
is_fraud_ip_address_1.columns = ['registration_week','is_fraud_ip_address_1']
is_fraud_ip_address_1 = list(is_fraud_ip_address_1['is_fraud_ip_address_1'])
grouped_df16['is_fraud_ip_address_1'] = is_fraud_ip_address_1
grouped_df16['is_fraud_ip_address_total'] = grouped_df16['is_fraud_ip_address_1'] + grouped_df16['is_fraud_ip_address_0']
grouped_df16['is_fraud_ip_address_1_sb_rate'] = grouped_df16['is_fraud_ip_address_1'] / grouped_df16['is_fraud_ip_address_total']
grouped_df16['is_fraud_ip_address_prev_wk_sb_rate'] = grouped_df16['is_fraud_ip_address_1_sb_rate'].shift()
grouped_df16['is_fraud_ip_address_sb_rate_pct_chg'] = grouped_df16['is_fraud_ip_address_prev_wk_sb_rate'].pct_change()
grouped_df = pd.merge(grouped_df, grouped_df16, left_on='registration_week', right_on='registration_week')

#fraud_ip_reg_hour
grouped_df17 = g.apply(lambda x: x[x['rule_fraud_ip_reg_hour'] == 0]['rule_fraud_ip_reg_hour'].count()).reset_index()
grouped_df17.columns = ['registration_week','fraud_ip_reg_hour_0']
fraud_ip_reg_hour_1 = g.apply(lambda x: x[x['rule_fraud_ip_reg_hour'] == 1]['rule_fraud_ip_reg_hour'].count()).reset_index()
fraud_ip_reg_hour_1.columns = ['registration_week','fraud_ip_reg_hour_1']
fraud_ip_reg_hour_1 = list(fraud_ip_reg_hour_1['fraud_ip_reg_hour_1'])
grouped_df17['fraud_ip_reg_hour_1'] = fraud_ip_reg_hour_1
grouped_df17['fraud_ip_reg_hour_total'] = grouped_df17['fraud_ip_reg_hour_1'] + grouped_df17['fraud_ip_reg_hour_0']
grouped_df17['fraud_ip_reg_hour_1_sb_rate'] = grouped_df17['fraud_ip_reg_hour_1'] / grouped_df17['fraud_ip_reg_hour_total']
grouped_df17['fraud_ip_reg_hour_prev_wk_sb_rate'] = grouped_df17['fraud_ip_reg_hour_1_sb_rate'].shift()
grouped_df17['fraud_ip_reg_hour_sb_rate_pct_chg'] = grouped_df17['fraud_ip_reg_hour_prev_wk_sb_rate'].pct_change()
grouped_df = pd.merge(grouped_df, grouped_df17, left_on='registration_week', right_on='registration_week')

#state_name_reg_hour
grouped_df18 = g.apply(lambda x: x[x['rule_state_name_reg_hour'] == 0]['rule_state_name_reg_hour'].count()).reset_index()
grouped_df18.columns = ['registration_week','state_name_reg_hour_0']
state_name_reg_hour_1 = g.apply(lambda x: x[x['rule_state_name_reg_hour'] == 1]['rule_state_name_reg_hour'].count()).reset_index()
state_name_reg_hour_1.columns = ['registration_week','state_name_reg_hour_1']
state_name_reg_hour_1 = list(state_name_reg_hour_1['state_name_reg_hour_1'])
grouped_df18['state_name_reg_hour_1'] = state_name_reg_hour_1
grouped_df18['state_name_reg_hour_total'] = grouped_df18['state_name_reg_hour_1'] + grouped_df18['state_name_reg_hour_0']
grouped_df18['state_name_reg_hour_1_sb_rate'] = grouped_df18['state_name_reg_hour_1'] / grouped_df18['state_name_reg_hour_total']
grouped_df18['state_name_reg_hour_prev_wk_sb_rate'] = grouped_df18['state_name_reg_hour_1_sb_rate'].shift()
grouped_df18['state_name_reg_hour_sb_rate_pct_chg'] = grouped_df18['state_name_reg_hour_prev_wk_sb_rate'].pct_change()
grouped_df = pd.merge(grouped_df, grouped_df18, left_on='registration_week', right_on='registration_week')

#origin_fraud_ip
grouped_df19 = g.apply(lambda x: x[x['rule_origin_fraud_ip'] == 0]['rule_origin_fraud_ip'].count()).reset_index()
grouped_df19.columns = ['registration_week','origin_fraud_ip_0']
origin_fraud_ip_1 = g.apply(lambda x: x[x['rule_origin_fraud_ip'] == 1]['rule_origin_fraud_ip'].count()).reset_index()
origin_fraud_ip_1.columns = ['registration_week','origin_fraud_ip_1']
origin_fraud_ip_1 = list(origin_fraud_ip_1['origin_fraud_ip_1'])
grouped_df19['origin_fraud_ip_1'] = origin_fraud_ip_1
grouped_df19['origin_fraud_ip_total'] = grouped_df19['origin_fraud_ip_1'] + grouped_df19['origin_fraud_ip_0']
grouped_df19['origin_fraud_ip_1_sb_rate'] = grouped_df19['origin_fraud_ip_1'] / grouped_df19['origin_fraud_ip_total']
grouped_df19['origin_fraud_ip_prev_wk_sb_rate'] = grouped_df19['origin_fraud_ip_1_sb_rate'].shift()
grouped_df19['origin_fraud_ip_sb_rate_pct_chg'] = grouped_df19['origin_fraud_ip_prev_wk_sb_rate'].pct_change()
grouped_df = pd.merge(grouped_df, grouped_df19, left_on='registration_week', right_on='registration_week')

#low_risk_rules_triggered
grouped_df20 = g.apply(lambda x: x[x['rule_low_risk_rules_triggered'] == 0]['rule_low_risk_rules_triggered'].count()).reset_index()
grouped_df20.columns = ['registration_week','low_risk_rules_triggered_0']
low_risk_rules_triggered_1 = g.apply(lambda x: x[x['rule_low_risk_rules_triggered'] == 1]['rule_low_risk_rules_triggered'].count()).reset_index()
low_risk_rules_triggered_1.columns = ['registration_week','low_risk_rules_triggered_1']
low_risk_rules_triggered_1 = list(low_risk_rules_triggered_1['low_risk_rules_triggered_1'])
grouped_df20['low_risk_rules_triggered_1'] = low_risk_rules_triggered_1
grouped_df20['low_risk_rules_triggered_total'] = grouped_df20['low_risk_rules_triggered_1'] + grouped_df20['low_risk_rules_triggered_0']
grouped_df20['low_risk_rules_triggered_1_sb_rate'] = grouped_df20['low_risk_rules_triggered_1'] / grouped_df20['low_risk_rules_triggered_total']
grouped_df20['low_risk_rules_triggered_prev_wk_sb_rate'] = grouped_df20['low_risk_rules_triggered_1_sb_rate'].shift()
grouped_df20['low_risk_rules_triggered_sb_rate_pct_chg'] = grouped_df20['low_risk_rules_triggered_prev_wk_sb_rate'].pct_change()
grouped_df = pd.merge(grouped_df, grouped_df20, left_on='registration_week', right_on='registration_week')

#risky_first_name
grouped_df21 = g.apply(lambda x: x[x['rule_risky_first_name'] == 0]['rule_risky_first_name'].count()).reset_index()
grouped_df21.columns = ['registration_week','risky_first_name_0']
risky_first_name_1 = g.apply(lambda x: x[x['rule_risky_first_name'] == 1]['rule_risky_first_name'].count()).reset_index()
risky_first_name_1.columns = ['registration_week','risky_first_name_1']
risky_first_name_1 = list(risky_first_name_1['risky_first_name_1'])
grouped_df21['risky_first_name_1'] = risky_first_name_1
grouped_df21['risky_first_name_total'] = grouped_df21['risky_first_name_1'] + grouped_df21['risky_first_name_0']
grouped_df21['risky_first_name_1_sb_rate'] = grouped_df21['risky_first_name_1'] / grouped_df21['risky_first_name_total']
grouped_df21['risky_first_name_prev_wk_sb_rate'] = grouped_df21['risky_first_name_1_sb_rate'].shift()
grouped_df21['risky_first_name_sb_rate_pct_chg'] = grouped_df21['risky_first_name_prev_wk_sb_rate'].pct_change()
grouped_df = pd.merge(grouped_df, grouped_df21, left_on='registration_week', right_on='registration_week')

#rule_signup_state_dev_ip_hour
grouped_df22 = g.apply(lambda x: x[x['rule_signup_state_dev_ip_hour'] == 0]['rule_signup_state_dev_ip_hour'].count()).reset_index()
grouped_df22.columns = ['registration_week','rule_signup_state_dev_ip_hour_0']
rule_signup_state_dev_ip_hour_1 = g.apply(lambda x: x[x['rule_signup_state_dev_ip_hour'] == 1]['rule_signup_state_dev_ip_hour'].count()).reset_index()
rule_signup_state_dev_ip_hour_1.columns = ['registration_week','rule_signup_state_dev_ip_hour_1']
rule_signup_state_dev_ip_hour_1 = list(rule_signup_state_dev_ip_hour_1['rule_signup_state_dev_ip_hour_1'])
grouped_df22['rule_signup_state_dev_ip_hour_1'] = rule_signup_state_dev_ip_hour_1
grouped_df22['rule_signup_state_dev_ip_hour_total'] = grouped_df22['rule_signup_state_dev_ip_hour_1'] + grouped_df22['rule_signup_state_dev_ip_hour_0']
grouped_df22['rule_signup_state_dev_ip_hour_1_sb_rate'] = grouped_df22['rule_signup_state_dev_ip_hour_1'] / grouped_df22['rule_signup_state_dev_ip_hour_total']
grouped_df22['rule_signup_state_dev_ip_hour_prev_wk_sb_rate'] = grouped_df22['rule_signup_state_dev_ip_hour_1_sb_rate'].shift()
grouped_df22['rule_signup_state_dev_ip_hour_sb_rate_pct_chg'] = grouped_df22['rule_signup_state_dev_ip_hour_prev_wk_sb_rate'].pct_change()
grouped_df = pd.merge(grouped_df, grouped_df22, left_on='registration_week', right_on='registration_week')

#rule_fraud_dev_bot_reg_hr
grouped_df23 = g.apply(lambda x: x[x['rule_fraud_dev_bot_reg_hr'] == 0]['rule_fraud_dev_bot_reg_hr'].count()).reset_index()
grouped_df23.columns = ['registration_week','rule_fraud_dev_bot_reg_hr_0']
rule_fraud_dev_bot_reg_hr_1 = g.apply(lambda x: x[x['rule_fraud_dev_bot_reg_hr'] == 1]['rule_fraud_dev_bot_reg_hr'].count()).reset_index()
rule_fraud_dev_bot_reg_hr_1.columns = ['registration_week','rule_fraud_dev_bot_reg_hr_1']
rule_fraud_dev_bot_reg_hr_1 = list(rule_fraud_dev_bot_reg_hr_1['rule_fraud_dev_bot_reg_hr_1'])
grouped_df23['rule_fraud_dev_bot_reg_hr_1'] = rule_fraud_dev_bot_reg_hr_1
grouped_df23['rule_fraud_dev_bot_reg_hr_total'] = grouped_df23['rule_fraud_dev_bot_reg_hr_1'] + grouped_df23['rule_fraud_dev_bot_reg_hr_0']
grouped_df23['rule_fraud_dev_bot_reg_hr_1_sb_rate'] = grouped_df23['rule_fraud_dev_bot_reg_hr_1'] / grouped_df23['rule_fraud_dev_bot_reg_hr_total']
grouped_df23['rule_fraud_dev_bot_reg_hr_prev_wk_sb_rate'] = grouped_df23['rule_fraud_dev_bot_reg_hr_1_sb_rate'].shift()
grouped_df23['rule_fraud_dev_bot_reg_hr_sb_rate_pct_chg'] = grouped_df23['rule_fraud_dev_bot_reg_hr_prev_wk_sb_rate'].pct_change()
grouped_df = pd.merge(grouped_df, grouped_df23, left_on='registration_week', right_on='registration_week')

#rule_device_linked_ge_15_accts
grouped_df24 = g.apply(lambda x: x[x['rule_device_linked_ge_15_accts'] == 0]['rule_device_linked_ge_15_accts'].count()).reset_index()
grouped_df24.columns = ['registration_week','rule_device_linked_ge_15_accts_0']
rule_device_linked_ge_15_accts_1 = g.apply(lambda x: x[x['rule_device_linked_ge_15_accts'] == 1]['rule_device_linked_ge_15_accts'].count()).reset_index()
rule_device_linked_ge_15_accts_1.columns = ['registration_week','rule_device_linked_ge_15_accts_1']
rule_device_linked_ge_15_accts_1 = list(rule_device_linked_ge_15_accts_1['rule_device_linked_ge_15_accts_1'])
grouped_df24['rule_device_linked_ge_15_accts_1'] = rule_device_linked_ge_15_accts_1
grouped_df24['rule_device_linked_ge_15_accts_total'] = grouped_df24['rule_device_linked_ge_15_accts_1'] + grouped_df24['rule_device_linked_ge_15_accts_0']
grouped_df24['rule_device_linked_ge_15_accts_1_sb_rate'] = grouped_df24['rule_device_linked_ge_15_accts_1'] / grouped_df24['rule_device_linked_ge_15_accts_total']
grouped_df24['rule_device_linked_ge_15_accts_prev_wk_sb_rate'] = grouped_df24['rule_device_linked_ge_15_accts_1_sb_rate'].shift()
grouped_df24['rule_device_linked_ge_15_accts_sb_rate_pct_chg'] = grouped_df24['rule_device_linked_ge_15_accts_prev_wk_sb_rate'].pct_change()
grouped_df = pd.merge(grouped_df, grouped_df24, left_on='registration_week', right_on='registration_week')


end_time3 = datetime.now()
print('Created grouped dfs for rule based analytics. ','Duration: {}'.format(end_time3 - start_time))


#only show pct change week over week columns
grouped_df_pct_chg = grouped_df.filter(regex='registration_week|pct_chg', axis=1)
grouped_df_pct_chg = grouped_df_pct_chg.fillna('0')

#convert all but week column to float
grouped_df_pct_chg['classify_handle_score_sb_rate_pct_chg'] = grouped_df_pct_chg['classify_handle_score_sb_rate_pct_chg'].astype('float')
grouped_df_pct_chg['prev_blocked_email_handle_sb_rate_pct_chg'] = grouped_df_pct_chg['prev_blocked_email_handle_sb_rate_pct_chg'].astype('float')
grouped_df_pct_chg['uncommon_domain_sb_rate_pct_chg'] = grouped_df_pct_chg['uncommon_domain_sb_rate_pct_chg'].astype('float')
grouped_df_pct_chg['disposable_domain_sb_rate_pct_chg'] = grouped_df_pct_chg['disposable_domain_sb_rate_pct_chg'].astype('float')
grouped_df_pct_chg['bot_score_bucket_sb_rate_pct_chg'] = grouped_df_pct_chg['bot_score_bucket_sb_rate_pct_chg'].astype('float')
grouped_df_pct_chg['is_emulator_sb_rate_pct_chg'] = grouped_df_pct_chg['is_emulator_sb_rate_pct_chg'].astype('float')
grouped_df_pct_chg['origin_sb_rate_pct_chg'] = grouped_df_pct_chg['origin_sb_rate_pct_chg'].astype('float')
grouped_df_pct_chg['likely_invalid_email_sb_rate_pct_chg'] = grouped_df_pct_chg['likely_invalid_email_sb_rate_pct_chg'].astype('float')
grouped_df_pct_chg['zip_sb_rate_pct_chg'] = grouped_df_pct_chg['zip_sb_rate_pct_chg'].astype('float')
grouped_df_pct_chg['signup_method_sb_rate_pct_chg'] = grouped_df_pct_chg['signup_method_sb_rate_pct_chg'].astype('float')
grouped_df_pct_chg['registration_hour_sb_rate_pct_chg'] = grouped_df_pct_chg['registration_hour_sb_rate_pct_chg'].astype('float')
grouped_df_pct_chg['email_nums_sb_rate_pct_chg'] = grouped_df_pct_chg['email_nums_sb_rate_pct_chg'].astype('float')
grouped_df_pct_chg['count_email_nums_sb_rate_pct_chg'] = grouped_df_pct_chg['count_email_nums_sb_rate_pct_chg'].astype('float')
grouped_df_pct_chg['state_name_sb_rate_pct_chg'] = grouped_df_pct_chg['state_name_sb_rate_pct_chg'].astype('float')
grouped_df_pct_chg['emails_are_bouncing_sb_rate_pct_chg'] = grouped_df_pct_chg['emails_are_bouncing_sb_rate_pct_chg'].astype('float')
grouped_df_pct_chg['is_fraud_device_id_sb_rate_pct_chg'] = grouped_df_pct_chg['is_fraud_device_id_sb_rate_pct_chg'].astype('float')
grouped_df_pct_chg['is_fraud_ip_address_sb_rate_pct_chg'] = grouped_df_pct_chg['is_fraud_ip_address_sb_rate_pct_chg'].astype('float')
grouped_df_pct_chg['fraud_ip_reg_hour_sb_rate_pct_chg'] = grouped_df_pct_chg['fraud_ip_reg_hour_sb_rate_pct_chg'].astype('float')
grouped_df_pct_chg['state_name_reg_hour_sb_rate_pct_chg'] = grouped_df_pct_chg['state_name_reg_hour_sb_rate_pct_chg'].astype('float')
grouped_df_pct_chg['origin_fraud_ip_sb_rate_pct_chg'] = grouped_df_pct_chg['origin_fraud_ip_sb_rate_pct_chg'].astype('float')
grouped_df_pct_chg['low_risk_rules_triggered_sb_rate_pct_chg'] = grouped_df_pct_chg['low_risk_rules_triggered_sb_rate_pct_chg'].astype('float')
grouped_df_pct_chg['risky_first_name_sb_rate_pct_chg'] = grouped_df_pct_chg['risky_first_name_sb_rate_pct_chg'].astype('float')
grouped_df_pct_chg['rule_signup_state_dev_ip_hour_sb_rate_pct_chg'] = grouped_df_pct_chg['rule_signup_state_dev_ip_hour_sb_rate_pct_chg'].astype('float')
grouped_df_pct_chg['rule_fraud_dev_bot_reg_hr_sb_rate_pct_chg'] = grouped_df_pct_chg['rule_fraud_dev_bot_reg_hr_sb_rate_pct_chg'].astype('float')
grouped_df_pct_chg['rule_device_linked_ge_15_accts_sb_rate_pct_chg'] = grouped_df_pct_chg['rule_device_linked_ge_15_accts_sb_rate_pct_chg'].astype('float')

print('Converted grouped df dtypes')

#merge pct_chg df columns to og df
df = pd.merge(df, grouped_df_pct_chg, left_on='registration_week', right_on='registration_week')

#layer 2 risk score changes based on previous weeks sb rate
#week over week rule sb rate change was positive: 25 points
m = df['classify_handle_score_sb_rate_pct_chg'].between(.01,.99)
df.loc[m, 'risk_score'] = df.loc[m, 'risk_score'] + 25
m = df['prev_blocked_email_handle_sb_rate_pct_chg'].between(.01,.99)
df.loc[m, 'risk_score'] = df.loc[m, 'risk_score'] + 25
m = df['uncommon_domain_sb_rate_pct_chg'].between(.01,.99)
df.loc[m, 'risk_score'] = df.loc[m, 'risk_score'] + 25
m = df['disposable_domain_sb_rate_pct_chg'].between(.01,.99)
df.loc[m, 'risk_score'] = df.loc[m, 'risk_score'] + 25
m = df['bot_score_bucket_sb_rate_pct_chg'].between(.01,.99)
df.loc[m, 'risk_score'] = df.loc[m, 'risk_score'] + 25
m = df['is_emulator_sb_rate_pct_chg'].between(.01,.99)
df.loc[m, 'risk_score'] = df.loc[m, 'risk_score'] + 25
m = df['origin_sb_rate_pct_chg'].between(.01,.99)
df.loc[m, 'risk_score'] = df.loc[m, 'risk_score'] + 25
m = df['likely_invalid_email_sb_rate_pct_chg'].between(.01,.99)
df.loc[m, 'risk_score'] = df.loc[m, 'risk_score'] + 25
m = df['zip_sb_rate_pct_chg'].between(.01,.99)
df.loc[m, 'risk_score'] = df.loc[m, 'risk_score'] + 25
m = df['signup_method_sb_rate_pct_chg'].between(.01,.99)
df.loc[m, 'risk_score'] = df.loc[m, 'risk_score'] + 25
m = df['registration_hour_sb_rate_pct_chg'].between(.01,.99)
df.loc[m, 'risk_score'] = df.loc[m, 'risk_score'] + 25
m = df['email_nums_sb_rate_pct_chg'].between(.01,.99)
df.loc[m, 'risk_score'] = df.loc[m, 'risk_score'] + 25
m = df['count_email_nums_sb_rate_pct_chg'].between(.01,.99)
df.loc[m, 'risk_score'] = df.loc[m, 'risk_score'] + 25
m = df['state_name_sb_rate_pct_chg'].between(.01,.99)
df.loc[m, 'risk_score'] = df.loc[m, 'risk_score'] + 25
m = df['emails_are_bouncing_sb_rate_pct_chg'].between(.01,.99)
df.loc[m, 'risk_score'] = df.loc[m, 'risk_score'] + 25
m = df['is_fraud_device_id_sb_rate_pct_chg'].between(.01,.99)
df.loc[m, 'risk_score'] = df.loc[m, 'risk_score'] + 25
m = df['is_fraud_ip_address_sb_rate_pct_chg'].between(.01,.99)
df.loc[m, 'risk_score'] = df.loc[m, 'risk_score'] + 25
m = df['fraud_ip_reg_hour_sb_rate_pct_chg'].between(.01,.99)
df.loc[m, 'risk_score'] = df.loc[m, 'risk_score'] + 25
m = df['state_name_reg_hour_sb_rate_pct_chg'].between(.01,.99)
df.loc[m, 'risk_score'] = df.loc[m, 'risk_score'] + 25
m = df['origin_fraud_ip_sb_rate_pct_chg'].between(.01,.99)
df.loc[m, 'risk_score'] = df.loc[m, 'risk_score'] + 25
m = df['low_risk_rules_triggered_sb_rate_pct_chg'].between(.01,.99)
df.loc[m, 'risk_score'] = df.loc[m, 'risk_score'] + 25
m = df['risky_first_name_sb_rate_pct_chg'].between(.01,.99)
df.loc[m, 'risk_score'] = df.loc[m, 'risk_score'] + 25
m = df['rule_signup_state_dev_ip_hour_sb_rate_pct_chg'].between(.01,.99)
df.loc[m, 'risk_score'] = df.loc[m, 'risk_score'] + 25
m = df['rule_fraud_dev_bot_reg_hr_sb_rate_pct_chg'].between(.01,.99)
df.loc[m, 'risk_score'] = df.loc[m, 'risk_score'] + 25
m = df['rule_device_linked_ge_15_accts_sb_rate_pct_chg'].between(.01,.99)
df.loc[m, 'risk_score'] = df.loc[m, 'risk_score'] + 25

#week over week rule sb rate change was negative: -25 points
m = df['classify_handle_score_sb_rate_pct_chg'] < 0
df.loc[m, 'risk_score'] = df.loc[m, 'risk_score'] - 25
m = df['prev_blocked_email_handle_sb_rate_pct_chg'] < 0
df.loc[m, 'risk_score'] = df.loc[m, 'risk_score'] - 25
m = df['uncommon_domain_sb_rate_pct_chg'] < 0
df.loc[m, 'risk_score'] = df.loc[m, 'risk_score'] - 25
m = df['disposable_domain_sb_rate_pct_chg'] < 0
df.loc[m, 'risk_score'] = df.loc[m, 'risk_score'] - 25
m = df['bot_score_bucket_sb_rate_pct_chg'] < 0
df.loc[m, 'risk_score'] = df.loc[m, 'risk_score'] - 25
m = df['is_emulator_sb_rate_pct_chg'] < 0
df.loc[m, 'risk_score'] = df.loc[m, 'risk_score'] - 25
m = df['origin_sb_rate_pct_chg'] < 0
df.loc[m, 'risk_score'] = df.loc[m, 'risk_score'] - 25
m = df['likely_invalid_email_sb_rate_pct_chg'] < 0
df.loc[m, 'risk_score'] = df.loc[m, 'risk_score'] - 25
m = df['zip_sb_rate_pct_chg'] < 0
df.loc[m, 'risk_score'] = df.loc[m, 'risk_score'] - 25
m = df['signup_method_sb_rate_pct_chg'] < 0
df.loc[m, 'risk_score'] = df.loc[m, 'risk_score'] - 25
m = df['registration_hour_sb_rate_pct_chg'] < 0
df.loc[m, 'risk_score'] = df.loc[m, 'risk_score'] - 25
m = df['email_nums_sb_rate_pct_chg'] < 0
df.loc[m, 'risk_score'] = df.loc[m, 'risk_score'] - 25
m = df['count_email_nums_sb_rate_pct_chg'] < 0
df.loc[m, 'risk_score'] = df.loc[m, 'risk_score'] - 25
m = df['state_name_sb_rate_pct_chg'] < 0
df.loc[m, 'risk_score'] = df.loc[m, 'risk_score'] - 25
m = df['emails_are_bouncing_sb_rate_pct_chg'] < 0
df.loc[m, 'risk_score'] = df.loc[m, 'risk_score'] - 25
m = df['is_fraud_device_id_sb_rate_pct_chg'] < 0
df.loc[m, 'risk_score'] = df.loc[m, 'risk_score'] - 25
m = df['is_fraud_ip_address_sb_rate_pct_chg'] < 0
df.loc[m, 'risk_score'] = df.loc[m, 'risk_score'] - 25
m = df['fraud_ip_reg_hour_sb_rate_pct_chg'] < 0
df.loc[m, 'risk_score'] = df.loc[m, 'risk_score'] - 25
m = df['state_name_reg_hour_sb_rate_pct_chg'] < 0
df.loc[m, 'risk_score'] = df.loc[m, 'risk_score'] - 25
m = df['origin_fraud_ip_sb_rate_pct_chg'] < 0
df.loc[m, 'risk_score'] = df.loc[m, 'risk_score'] - 25
m = df['low_risk_rules_triggered_sb_rate_pct_chg'] < 0
df.loc[m, 'risk_score'] = df.loc[m, 'risk_score'] - 25
m = df['risky_first_name_sb_rate_pct_chg'] < 0
df.loc[m, 'risk_score'] = df.loc[m, 'risk_score'] - 25
m = df['rule_signup_state_dev_ip_hour_sb_rate_pct_chg'] < 0
df.loc[m, 'risk_score'] = df.loc[m, 'risk_score'] - 25
m = df['rule_fraud_dev_bot_reg_hr_sb_rate_pct_chg'] < 0
df.loc[m, 'risk_score'] = df.loc[m, 'risk_score'] - 25
m = df['rule_device_linked_ge_15_accts_sb_rate_pct_chg'] < 0
df.loc[m, 'risk_score'] = df.loc[m, 'risk_score'] - 25
print('Applied rule score changes')

#convert some last remaining columns to the right dtype
#google big query had an issue with pandas dates
# df['email_nums'] = df['email_nums'].astype('int')
#df['first_login_date'] = df['first_login_date'].astype('str')
df['registration_day'] = df['registration_day'].astype('str')
df['registration_week'] = df['registration_week'].astype('str')
df['registration_date'] = df['registration_date'].astype('str')
df['registration_hour'] = df['registration_hour'].astype('str')
# df['risk_score_bin'] = df['risk_score_bin'].astype('str')
df['score_bin'] = df['score_bin'].astype('str')
df['ua_part1'] = df['ua_part1'].astype('str')
#df['first_login_date'] = df['first_login_date'].astype('str')
df['user_agent'] = df['user_agent'].astype('str')
df['state_name'] = df['state_name'].astype('str')
df['zip'] = df['zip'].astype('str')
df['firewall_bot_score'] = df['firewall_bot_score'].astype('str')
# df['device_fingerprint'] = df['device_fingerprint'].astype('str')
df['first_name'] = df['first_name'].astype('str')
# df['login_source'] = df['login_source'].astype('str')
# df['action'] = df['action'].astype('str')
df['device_is_blacklisted'] = df['device_is_blacklisted'].astype('str')
df['email'] = df['email'].astype('str')
df['app_version'] = df['app_version'].astype('str')
df['app_build'] = df['app_build'].astype('str')
df['device_type'] = df['device_type'].astype('str')
df['iphone_type'] = df['iphone_type'].astype('str')
df['os'] = df['os'].astype('str')
df['os_type'] = df['os_type'].astype('str')
df['os_version'] = df['os_version'].astype('str')
df['lang'] = df['lang'].astype('str')
df['registration_hour'] = df['registration_hour'].astype('str')
df['email_handle'] = df['email_handle'].astype('str')
df['email_domain'] = df['email_domain'].astype('str')
df['email_nums'] = df['email_nums'].astype('str')
df['sb_email_handle'] = df['sb_email_handle'].astype('str')
print('Convert last remaining cols for BigQuery ingest')

#*************ML starts here**************

from ast import literal_eval
import math
import re

import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

from difflib import SequenceMatcher
from collections import Counter
from itertools import groupby

from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import VotingClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
# import xgboost as xgb

# names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
#          "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
#          "Naive Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]

# from keras import layers
# from keras import models
# from keras import optimizers
# from keras import losses
# from keras import metrics

#create ml df out of low score rows from df
# ml_df = df[df['risk_score_bin'].isin(['-50','0','25'])] #change this back to only run ml on low scores
ml_df =  df[['user_id','registration_day'
            ,'rule_prev_blocked_email_handle','rule_uncommon_domain','rule_disposable_domain','rule_bot_score_bucket','rule_is_emulator' 
            ,'rule_origin','rule_likely_invalid_email','rule_zip','rule_signup_method','rule_registration_hour','rule_email_nums' 
            ,'rule_count_email_nums','rule_state_name','rule_emails_are_bouncing','rule_is_fraud_device_id','rule_is_fraud_ip_address' 
            ,'rule_fraud_ip_reg_hour','rule_state_name_reg_hour','rule_origin_fraud_ip','rule_low_risk_rules_triggered','rule_risky_first_name' 
            ,'rule_fraud_dev_bot_reg_hr','rule_device_linked_ge_15_accts'
            ,'risk_score','is_two_names','prev_blocked_email_handle','uncommon_domain','likely_invalid_email','disposable_domain'
            ,'bot_score_bucket','is_emulator','firewall_bot_score','registration_hour','count_email_nums','is_fraud_device_id'
#              ,'emails_are_bouncing'
            ,'classify_handle_score','classify_handle_score_rule','rule_signup_state_dev_ip_hour'
            ,'is_softblocked_tf']]

#convert ml_df object cols:
# ml_df['risk_score_bin'] = ml_df['risk_score_bin'].replace(np.nan,0)
# ml_df['risk_score_bin'] = ml_df['risk_score_bin'].astype(int)
ml_df['firewall_bot_score'] = ml_df['firewall_bot_score'].astype(float)
ml_df['registration_hour'] = ml_df['registration_hour'].astype(float)
ml_df = ml_df.fillna(0)

print('Machine Learning df created')
print(ml_df.shape)
# display(ml_df.head(2))
# print(ml_df.columns)

ml_df_array = ml_df[ml_df.columns[2:39]].values
# ml_df_array

#Create Test and Train arrays on low risk scores df
#if you add more columns to the dataset
#make sure you adjust the X and y column ranges
X = ml_df[ml_df.columns[2:39]].values
y = ml_df[ml_df.columns[-1]].values

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.90)

#print out train array shapes
# print('X_train shape: \n', X_train.shape)
# print('X_test shape: \n', X_test.shape)
# print('y_train shape: \n', y_train.shape)
# print('y_test shape: \n', y_test.shape)
# print('')
# print('x_train:', X_train)
# print('y_train:', y_train)
# print('x_test:', X_test)
# print('y_test:', y_test)

print('ML train test splits completed')

#run ml on org df
#classify models
# nb_model = GaussianNB()
# lr_model = LogisticRegression(max_iter=1000)
# dt_model = DecisionTreeClassifier(max_depth=5)
rf_model = RandomForestClassifier()
# # knn_model = KNeighborsClassifier()
# svm_model = LinearSVC(C=0.0001)
# bg_model = BaggingClassifier(DecisionTreeClassifier(),max_samples=0.5,max_features=1.0,n_estimators=10)
adb_model = AdaBoostClassifier(DecisionTreeClassifier(min_samples_split=10,max_depth=4),n_estimators=10,learning_rate=0.6)
# vote_model = VotingClassifier(estimators=[('nb',nb_model),('lr',lr_model),('rf',rf_model),('dt',dt_model)],voting='hard')
# svc_gamma_model = SVC(gamma=2, C=1)
# gaussian_model = GaussianProcessClassifier(1.0 * RBF(1.0))
# # RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
# nn_model = MLPClassifier(alpha=1, max_iter=1000)
# quad_discrim_model = QuadraticDiscriminantAnalysis()

#fitting model with prediction data and telling it my target
# nb_model.fit(X_train, y_train)
# print('nb_model fit complete')
# lr_model.fit(X_train, y_train)
# print('lr_model fit complete')
# dt_model.fit(X_train, y_train)
# print('dt_model fit complete')
rf_model.fit(X_train, y_train)
print('rf_model fit complete')
# # knn_model.fit(X_train, y_train)
# svm_model.fit(X_train, y_train)
# print('svm_model fit complete')
# bg_model.fit(X_train, y_train)
# print('bg_model fit complete')
adb_model.fit(X_train, y_train)
print('adb_model fit complete')
# vote_model.fit(X_train, y_train)
# print('vote_model fit complete')
# svc_gamma_model.fit(X_train, y_train)
# print('svc_gamma_model fit complete')
# gaussian_model.fit(X_train, y_train)
# print('gaussian_model fit complete')
# nn_model.fit(X_train, y_train)
# print('nn_model fit complete')
# quad_discrim_model.fit(X_train, y_train)
# print('quad_discrim_model fit complete')


print('ML models fitted on train data')

#test model against test data (actual data without target)
# nb_pred = list(nb_model.predict(ml_df_array))
# lr_pred = list(lr_model.predict(ml_df_array))
# dt_pred = list(dt_model.predict(ml_df_array))
rf_pred = list(rf_model.predict(ml_df_array))
# # knn_pred = list(knn_model.predict(ml_df_array))
# svm_pred = list(svm_model.predict(ml_df_array))
# bg_pred = list(bg_model.predict(ml_df_array))
adb_pred = list(adb_model.predict(ml_df_array))
# vote_pred = list(vote_model.predict(ml_df_array))
# svc_gamma_pred = list(svc_gamma_model.predict(ml_df_array))
# gaussian_pred = list(gaussian_model.predict(ml_df_array))
# nn_pred = list(nn_model.predict(ml_df_array))
# quad_discrim_pred = list(quad_discrim_model.predict(ml_df_array))


print('ML models predictions on ml_df_array completed')

#run ml on the ml_df
#create random forest column
# ml_df['nb_pred'] = nb_pred
# ml_df['lr_pred'] = lr_pred
# ml_df['dt_pred'] = dt_pred
ml_df['rf_pred'] = rf_pred
# # ml_df['knn_pred'] = knn_pred
# ml_df['svm_pred'] = svm_pred
# ml_df['bg_pred'] = bg_pred
ml_df['adb_pred'] = adb_pred
# ml_df['vote_pred'] = vote_pred
# ml_df['svc_gamma_pred'] = svc_gamma_pred
# ml_df['gaussian_pred'] = gaussian_pred
# ml_df['nn_pred'] = nn_pred
# ml_df['quad_discrim_pred'] = quad_discrim_pred


#convert prediction cols to ints
# ml_df['nb_pred'] = ml_df['nb_pred'].astype(int)
# ml_df['lr_pred'] = ml_df['lr_pred'].astype(int)
# ml_df['dt_pred'] = ml_df['dt_pred'].astype(int)
ml_df['rf_pred'] = ml_df['rf_pred'].astype(int)
# # ml_df['knn_pred'] = ml_df['knn_pred'].astype(int)
# ml_df['svm_pred'] = ml_df['svm_pred'].astype(int)
# ml_df['bg_pred'] = ml_df['bg_pred'].astype(int)
ml_df['adb_pred'] = ml_df['adb_pred'].astype(int)
# ml_df['vote_pred'] = ml_df['vote_pred'].astype(int)
# ml_df['svc_gamma_pred'] = ml_df['svc_gamma_pred'].astype(int)
# ml_df['gaussian_pred'] = ml_df['gaussian_pred'].astype(int)
# ml_df['nn_pred'] = ml_df['nn_pred'].astype(int)
# ml_df['quad_discrim_pred'] = ml_df['quad_discrim_pred'].astype(int)


print('ML columns added to ml_df')

#filter out only predicted fraud
# org_df = org_df[org_df['is_likely_bad'] == 1]

print('ml_df shape: ', ml_df.shape)
# ml_df.head()


#### place holder for code to automatically grab best performing classifiers and put them in the below function
##manual for now


#create new column for diff combined ml prediction scores
def ml_pred_score(row):
    if row['rf_pred'] == 1:
        return 1
    elif row['adb_pred'] == 1:
        return 1
    elif row['is_softblocked_tf'] == 1:
        return 1
    else:
        return 0

ml_df['ml_pred_score'] = ml_df.apply(lambda row: ml_pred_score(row), axis = 1)
ml_df['ml_pred_score'] = ml_df['ml_pred_score'].astype('int')

#create duplicate ml prediction column with just ml predictions
#this can change - add more classifiers in conditions if you want
def ml_predictions(row):
    if row['rf_pred'] == 1:
        return 1
    elif row['adb_pred'] == 1:
        return 1
    else:
        return 0

ml_df['ml_predictions'] = ml_df.apply(lambda row: ml_predictions(row), axis = 1)
ml_df['ml_predictions'] = ml_df['ml_predictions'].astype('int')

#replace is_softblocked column with machine learning is softblocked column
#drop prediction columns so the concatinating dfs have same cols
# ml_df.drop('is_softblocked_tf', axis=1, inplace=True)
# ml_df.drop('nb_pred', axis=1, inplace=True)
# ml_df.drop('lr_pred', axis=1, inplace=True)
# ml_df.drop('dt_pred', axis=1, inplace=True)
# ml_df.drop('rf_pred', axis=1, inplace=True)
# # ml_df.drop('knn_pred', axis=1, inplace=True)
# ml_df.drop('svm_pred', axis=1, inplace=True)
# ml_df.drop('bg_pred', axis=1, inplace=True)
# ml_df.drop('adb_pred', axis=1, inplace=True)
# ml_df.drop('vote_pred', axis=1, inplace=True)

# ml_df=ml_df.rename(columns = {'ml_pred_score':'is_softblocked_tf'})

# print('ML columns dropped from ml_df and softblocked col renamed')

#show the actual userids after machine learning
#original df
# df_org = df[~df['risk_score_bin'].isin(['-50','0','25'])]
# df_org =  df_org[['user_id','registration_day'
#             ,'rule_prev_blocked_email_handle','rule_uncommon_domain','rule_disposable_domain','rule_bot_score_bucket','rule_is_emulator' 
#             ,'rule_origin','rule_likely_invalid_email','rule_zip','rule_signup_method','rule_registration_hour','rule_email_nums' 
#             ,'rule_count_email_nums','rule_state_name','rule_emails_are_bouncing','rule_is_fraud_device_id','rule_is_fraud_ip_address' 
#             ,'rule_fraud_ip_reg_hour','rule_state_name_reg_hour','rule_origin_fraud_ip','rule_low_risk_rules_triggered','rule_risky_first_name' 
#             ,'rule_fraud_dev_bot_reg_hr','risk_score','risk_score_bin','is_two_names','prev_blocked_email_handle','uncommon_domain','likely_invalid_email' 
#             ,'disposable_domain','bot_score_bucket','is_emulator','firewall_bot_score','registration_hour','count_email_nums'
#             ,'is_fraud_device_id','emails_are_bouncing' ,'classify_handle_score','classify_handle_score_rule','rule_signup_state_dev_ip_hour'
#             ,'is_softblocked_tf']]
# print('df_org shape: ', df_org.shape)
# df_org.head(2)

#ml_df
# m = ml_df['is_softblocked_tf'] == '1'
# ml_df.loc[m, 'risk_score'] = ml_df.loc[m, 'risk_score'] == '2000' #check diff numbers
# print('ml_df shape: ', ml_df.shape)

#manually change risk scores for ml predicted fraud
ml_df = ml_df.rename(columns = {'risk_score':'risk_score_rules'})
def risk_score_ml(row):
    if row['ml_pred_score'] == 1:
        return row['risk_score_rules'] + 1000
    else:
        return row['risk_score_rules']
ml_df['risk_score_ml'] = ml_df.apply(lambda row: risk_score_ml(row), axis = 1)

print('Changed ml scores')

# #concat dfs
# df_final = pd.concat([df_org,ml_df], ignore_index=True)
# print('df_final shape: ', df_final.shape)

# #convert 0s and dtypes
# df_final['is_softblocked_tf'] = df_final['is_softblocked_tf'].astype('int')

# print('ML df concatenated with Orginal df')

#***************Create da df here****************

#rename new prediction column and old softblocked column
ml_df = ml_df.rename(columns = {'is_softblocked_tf':'is_softblocked_tf_rules'})
ml_df = ml_df.rename(columns = {'ml_pred_score':'is_softblocked_tf'})

#import fraud function
def data_analysis(df, field, granularity = 1,xlim_sb = None, ylim_sb = None, xlim_total = None, ylim_total = None, figsize = (26,6)
                  ,minrate=None,maxrate=None,maxcount=None):
    if(granularity == 1):
        field_table = pd.crosstab(df[field],df.is_softblocked_tf) #change back to is_softblocked_tf
    else:
        field_table = pd.crosstab(eval('(df[field]'+'/'+str(granularity) +').astype(int)*' + str(granularity)),df.is_softblocked_tf)
    field_table = field_table.reset_index(0)
    field_table = field_table.rename_axis('', axis = 1)
    field_table['is_softblocked_rate'] = field_table[1]/(field_table[0] + field_table[1])*100
    field_table['total_count'] = field_table[0] + field_table[1]
    field_table['percent_of_total'] = field_table['total_count']/sum(field_table['total_count']) * 100
    if minrate:
        field_table = field_table[field_table['is_softblocked_rate'] >= minrate]
    if maxrate:
        field_table = field_table[field_table['is_softblocked_rate'] <= maxrate]
    if maxcount:
        field_table = field_table[field_table['total_count'] >= maxcount]
    field_table = field_table.sort_values(by=field,ascending=True)
    return field_table

#create new risk score bins col
bins = [-100, -50, 0, 25, 50, 75, 100, 125, 150, 175, 200, 250, 300, 350, 400, 450, 500, 550
        , 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050, 2000]
labels = ['-100', '-50', '0', '25', '50', '75', '100', '125', '150', '175', '200', '250'
          , '300', '350', '400', '450', '500', '550', '600', '650', '700', '750', '800', '850', '900', '950', '1000', '1050']
# ml_df['risk_score_bin'] = pd.cut(ml_df['risk_score_ml'], bins=bins, labels=labels) #changed to ml_df from df_final
ml_df['risk_score'] = pd.cut(ml_df['risk_score_ml'], bins=bins, labels=labels) #changed to ml_df from df_final

#score evaluation field
field = 'risk_score'

#evaluate column/feature
da = data_analysis(df=ml_df,field=field)

#weird way to sort b/c plotly is dumb
#was all risk_score_bin
da['risk_score'] = da['risk_score'].astype(int)
da = da.sort_values(by='risk_score',ascending=True)
da['risk_score'] = da['risk_score'].astype(str)
# df['risk_score_bin'] = df['risk_score_bin'].astype(int)
# df['risk_score_bin'] = df['risk_score_bin'].astype(str)
df['os_version'] = df['os_version'].astype('str')
df['risk_score'] = df['risk_score'].fillna(0.0)


#change col names
#was risk_score_bin
da.columns = ['risk_score','not_softblocked','softblocked','is_softblocked_rate','total_count','percent_of_total']
print('da df completed')

end_time = datetime.now()
print('Script Completed. ','Query Duration: {}'.format(end_time - start_time))
print('df shape: ', df.shape)
print('unique users: ', df.user_id.nunique())
print('df size in gb: ', round(df.memory_usage(deep=True).sum() / 1000000000,2),'gb')

display('original df: ', df.head(2))
display('ml_df: ', ml_df.head())
display('agg_df_day: ', agg_df_day.head())
display('da: ', da.head())
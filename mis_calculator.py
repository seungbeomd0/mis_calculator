import numpy as np
import pandas as pd
import streamlit as st


def get_ev_count(
    list_ev_total: list,
    list_ev_right: list
):
    
    ev_total1, ev_total2, ev_total3, ev_total4, ev_total5 = list_ev_total 
    ev_total = np.sum(list_ev_total)
    ev_right1, ev_right2, ev_right3, ev_right4, ev_right5 = list_ev_right 
    ev_right = np.sum(list_ev_right)
    
    ev_count = pd.DataFrame({'ev_total': [ev_total], 'ev_total_1': [ev_total1], 'ev_total_2': [ev_total2], 
                             'ev_total_3': [ev_total3], 'ev_total_4': [ev_total4], 'ev_total_5': [ev_total5],
                             'ev_right': [ev_right], 'ev_right_1': [ev_right1], 'ev_right_2': [ev_right2], 
                             'ev_right_3': [ev_right3], 'ev_right_4': [ev_right4], 'ev_right_5': [ev_right5]})
    return ev_count

def get_acm_count(
    num_eval: int,
    list_ev_total: list,
    list_ev_right: list
):
    acm_count = pd.DataFrame()
    for i in np.arange(0, num_eval, 1):
        ev_count = get_ev_count(list_ev_total, list_ev_right)
        ev_count.insert(0, 'evalid', i+1)
        acm_count = pd.concat([acm_count, ev_count], axis = 0)
    
    acm_count_temp = acm_count.iloc[:, 1:].cumsum(axis = 0)
    acm_count_temp.rename(columns = {'ev_total': 'acm_total', 'ev_right': 'acm_right'}, inplace = True)
    acm_count_temp.rename(columns = {f"ev_total_{i}" : f"acm_total_{i}" for i in range(1,6)}, inplace = True)
    acm_count_temp.rename(columns = {f"ev_right_{i}" : f"acm_right_{i}" for i in range(1,6)}, inplace = True)

    acm_count = pd.concat([acm_count, acm_count_temp], axis = 1)

    return acm_count

def calculation_sMISitg_v2(
    df_simulation_eval_seq,
    weights_bfr_ev_good = [0.7, 0.3], 
    weights_bfr_ev_bad = [0.8, 0.2],
    ev_goodweight = [1, 2.3, 3.1, 4.0, 4.7], 
    ev_badweight = [0, 0, 0.5, 1.2, 2.1],
    param_convergence = 1.5, 
    param_rightrate = 0.5, 
    weight_question = 4
):
    
        weight_bfr_good, weight_ev_good = weights_bfr_ev_good
        weight_bfr_bad, weight_ev_bad = weights_bfr_ev_bad
            
        if isinstance(weight_question, list) == False:
            weight_question = [weight_question for x in np.arange(5)]
            
        #ev_score를 계산한다.
        df_simulation_eval_seq['ev_false_1'] = df_simulation_eval_seq['ev_total_1'] - df_simulation_eval_seq['ev_right_1']
        df_simulation_eval_seq['ev_false_2'] = df_simulation_eval_seq['ev_total_2'] - df_simulation_eval_seq['ev_right_2']
        df_simulation_eval_seq['ev_false_3'] = df_simulation_eval_seq['ev_total_3'] - df_simulation_eval_seq['ev_right_3']
        df_simulation_eval_seq['ev_false_4'] = df_simulation_eval_seq['ev_total_4'] - df_simulation_eval_seq['ev_right_4']
        df_simulation_eval_seq['ev_false_5'] = df_simulation_eval_seq['ev_total_5'] - df_simulation_eval_seq['ev_right_5']
        
        df_simulation_eval_seq['ev_right_value'] = df_simulation_eval_seq.apply(lambda row: sum([row[f'ev_right_{i+1}'] * ev_goodweight[i] for i in range(5)]), axis = 1)
        df_simulation_eval_seq['ev_false_value'] = df_simulation_eval_seq.apply(lambda row: sum([row[f'ev_false_{i+1}'] * ev_badweight[i] for i in range(5)]), axis = 1)
        df_simulation_eval_seq['ev_value'] = df_simulation_eval_seq['ev_right_value'] + df_simulation_eval_seq['ev_false_value']
        df_simulation_eval_seq['ev_total_value'] = df_simulation_eval_seq.apply(lambda row: sum([row[f'ev_total_{i+1}'] * weight_question[i] for i in range(5)]), axis = 1)

        df_simulation_eval_seq['ev_score'] = ((df_simulation_eval_seq['ev_value'] + param_convergence * (param_rightrate/np.sqrt(df_simulation_eval_seq['ev_total'])))
                                            / (df_simulation_eval_seq['ev_total_value'] + param_convergence))
        df_simulation_eval_seq.drop(columns = ['ev_false_1', 'ev_false_2', 'ev_false_3', 'ev_false_4', 'ev_false_5',
                                                'ev_right_value', 'ev_false_value', 'ev_total_value'], inplace = True)

        list_itg_score = [] #임시 리스트 
        make_log = []
        for idx, row in df_simulation_eval_seq.iterrows():
            
            if row['evalid'] == 1: #첫번째 eval이면 직전 itg_score는 현재 ev_score로 대체
                itg_score_bfr = row['ev_score']
                if row['ev_score'] >= itg_score_bfr: #정답 문제 배점 합 >= 틀린 문제 배점 합인 경우
                    itg_score = weight_bfr_good * itg_score_bfr + weight_ev_good * row['ev_score'] 
                else:
                    itg_score = weight_bfr_bad * itg_score_bfr + weight_ev_bad * row['ev_score']  
            else: 
                itg_score_bfr = itg_score
                if row['ev_score'] >= itg_score_bfr: #정답 문제 배점 합 >= 틀린 문제 배점 합인 경우
                    itg_score = weight_bfr_good * itg_score_bfr + weight_ev_good * row['ev_score'] 
                else:
                    itg_score = weight_bfr_bad * itg_score_bfr + weight_ev_bad * row['ev_score']  
                
            list_itg_score.append(itg_score)
            make_log.append("itg_bfr {} + ev_now {} = {}".format(round(itg_score_bfr, 3), 
                                                                 round(row['ev_score'], 3), 
                                                                 round(itg_score, 3)))
        
        df_simulation_eval_seq['itg_score'] = list_itg_score
        df_simulation_eval_seq['make_log'] = make_log
        
        return df_simulation_eval_seq
    
def tag_label(itg_score):
    if itg_score >= 0.65:
        label = '매우우수'
    elif itg_score >= 0.55:
        label = '우수'
    elif itg_score >= 0.35:
        label = '보통(이전:양호)'
    elif itg_score >= 0.25:
        label = '다소미흡(이전:보통)'
    else:
        label = '미흡'
    return label

# Streamlit 코드 시작
st.title("임시 개념이해도 계산기... 소영님 파이팅")

st.write("Please input the following values:")

# 유저 인풋 받기
TC2 = st.number_input("TC2", value=0.0)
TC3 = st.number_input("TC3", value=0.0)
TC4 = st.number_input("TC4", value=0.0)
RC2 = st.number_input("RC2", value=0.0)
RC3 = st.number_input("RC3", value=0.0)
RC4 = st.number_input("RC4", value=0.0)

# 계산 버튼 추가
if st.button("Calculate"):
    itg_score = calculation_sMISitg_v2((get_acm_count(1, [0,TC2,TC3,TC4,0], [0,RC2,RC3,RC4,0])))['itg_score'].values[0]
    label = tag_label(itg_score)
    st.success(f"이해도 점수: {np.round(itg_score,3)} | 등급: {label}")
    


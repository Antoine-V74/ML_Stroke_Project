import pandas as pd

def labelling(df):
    motor_tests = [
        "Fugl.Meyer_affected_TOTAL",
        "P.G_affected_FIST_mean",
        "B.B_blocks_affected_hand",
        "Purdue_affected_hand"
    ]

    qol_tests = [
        "mRS",
        "Barthel"
    ]

    attention_tests = [    
        "TAP_alert_without_warning_RT",
        "TAP_alert_with_warning_RT",
        "TAP_divided_attention_single_condition_Auditive_RT", 
        "TAP_divided_attention_single_condition_Visual_RT", 
        "TAP_divided_attention_both_condition_Auditive_RT",
        "TAP_divided_attention_both_condition_Visual_RT",
        "Bells_omissions_total.1",
        "CTM_A_time"
    ]

    executive_tests = [
        "Bi.manual_coordination_corrected",
        "FAB_TOT", 
        "AST_unaffected_TOTAL", 
        "CERAD_copy_TOTAL",
        "Stroop_interference_time",
        "Digit_sequencing_TOTAL",
        "Digit_backward_TOTAL",
        "Corsi_backward_TOTAL",
        "CTM_B_time"
    ]

    memory_tests = [
        "Corsi_forward_TOTAL",
        "Digit_forward_TOTAL"
    ]

    sensory_test = [
        "RASP_TOTAL_unaffected"
    ]

    Language_tests = [
        "Fluency_phon_final_score",
        "Fluency_sem_final_score",
        "LAST_TOTAL"
    ]

    Neglect_tests = [
        "Line_bissec_20cm",
        "Line_bissec_.5cm",
        "Bells_omissions_L.R"
    ]
    
    columns_for_labelisation =  motor_tests + qol_tests + attention_tests + executive_tests + memory_tests + sensory_test + Language_tests + Neglect_tests
    filtered_df = df[columns_for_labelisation]

    thresholds = {
        "Fugl.Meyer_affected_TOTAL": 50,
        "P.G_affected_FIST_mean": 18,
        "B.B_blocks_affected_hand": 40,
        "Purdue_affected_hand": 12,
        "mRS": 1,
        "Barthel": 90, 
        "TAP_alert_without_warning_RT": 400,
        "TAP_alert_with_warning_RT" : 300,
        "TAP_divided_attention_single_condition_Auditive_RT" : 450, 
        "TAP_divided_attention_single_condition_Visual_RT" : 400, 
        "TAP_divided_attention_both_condition_Auditive_RT" : 550,
        "TAP_divided_attention_both_condition_Visual_RT" : 500,
        "Bells_omissions_total.1" : 6,
        "CTM_A_time": 60,
        "Bi.manual_coordination_corrected": 85,
        "FAB_TOT" : 16, 
        "AST_unaffected_TOTAL" : 15, 
        "CERAD_copy_TOTAL" : 9,
        "Stroop_interference_time": 90,
        "Digit_sequencing_TOTAL" : 6,
        "Digit_backward_TOTAL" : 4,
        "Corsi_backward_TOTAL" : 4,
        "CTM_B_time" : 120,
        "Corsi_forward_TOTAL" : 5,
        "Digit_forward_TOTAL" : 6,
        "RASP_TOTAL_unaffected" : 60,
        "Fluency_phon_final_score" : 15,
        "Fluency_sem_final_score" : 20,
        "LAST_TOTAL" : 40,
        "Line_bissec_20cm" : 2,
        "Line_bissec_.5cm" : 1,
        "Bells_omissions_L.R" : 2
    }

    # Initialize labels as a DataFrame
    labels = filtered_df.copy()

    # Step 2: Calculate votes
    for col, threshold in thresholds.items():
        t4_values = filtered_df.get(col)
        labels[col] = (t4_values >= threshold).astype(int)

    # Step 3: Apply majority vote
    labels["Recovered"] = labels.iloc[:, 1:].sum(axis=1) >= (len(thresholds) / 2)

    # Convert labels to 0 or 1
    labels["Recovered"] = labels["Recovered"].astype(int)

    return labels["Recovered"]

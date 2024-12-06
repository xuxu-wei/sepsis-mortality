
def load_data(data, outcome_ix=0):
    features, _, _, treatment, outcomes = get_ite_features()
    X = data[features].copy() # 患者特征
    W = data[treatment] # 治疗分配
    y = data[outcomes[outcome_ix]] # 主要结局指标 28d死亡率
    return X, W, y 

def get_ite_model_vars():
    cate_vars = [var for var in ite_var_dict if ite_var_dict[var]=='category']
    cont_vars = [var for var in ite_var_dict if ite_var_dict[var]=='continuous']
    outcomes = ['28d_mortality', 'in_hospital_mortality','7d_septic_shock']
    treatment = 'XBJ_intervention'
    return cate_vars, cont_vars, treatment, outcomes

def get_ite_features():
    cate_vars, cont_vars, treatment, outcomes = get_ite_model_vars()
    cate_features = [var for var in cate_vars if not var in [treatment, *outcomes]]
    cont_features = [var for var in cont_vars if not var in [treatment, *outcomes]]
    features = [var for var in list(ite_var_dict.keys()) if not var in [treatment, *outcomes]]
    return features, cate_features, cont_features, treatment, outcomes


ite_var_dict = {
"XBJ_intervention":"category",
"age":"continuous",
"sex":"category",
"BMI":"continuous",

"primary_infection_site_lung":"category",
"primary_infection_site_abdo":"category",
"primary_infection_site_uri":"category",
"primary_infection_site_skin":"category",
"primary_infection_site_brain":"category",
"primary_infection_site_blood":"category",

"pathogen_test":"category",
"Gram-_infect":"category",
"Gram+_infect":"category",
"Fungi_infect":"category",
"Gram_neg_resist":"category",
"Gram_pos_resist":"category",
"Fungi_resist":"category",
"multidrug_resist":"category",

"temperature":"continuous",
"heart_rate":"continuous",
"respiratory_rate":"continuous",
# "SBP":"continuous",
# "DBP":"continuous",
"MAP":"continuous",

# "SOFA_respiration":"category",
# "SOFA_coagulation":"category",
# "SOFA_liver":"category",
# "SOFA_cardio":"category",
# "SOFA_cns":"category",
# "SOFA_renal":"category",
# "SOFA":"category",
# "APACHE_II":"category",
# "DIC-score":"category",

"septic_shock":"category",
"Respiratory_Support":"category",
"CCRT":"category",
"nutri_support":"category",
# "nutri_support_enteral":"category",
# "nutri_support_parenteral":"category",

"RBC":"continuous",
"Hb":"continuous",
"WBC":"continuous",
"NE%":"continuous",
"LYM%":"continuous",
"PLT":"continuous",
# "HCT":"continuous",
"ALT":"continuous",
"AST":"continuous",
"STB":"continuous",
"BUN":"continuous",
"Scr":"continuous",
"Glu":"continuous",
"K+":"continuous",
"Na+":"continuous",
"Fg":"continuous",
"PT":"continuous",
"APTT":"continuous",
# "D-Dimer":"continuous",
"CRP":"continuous",
# "PCT":"continuous",
"PH":"continuous",
"PaO2":"continuous",
"PaO2/FiO2":"continuous",
"Lac":"continuous",
"PaCO2":"continuous",
# "HCO3-":"continuous",

"28d_mortality":"category",
"in_hospital_mortality":"category",
"7d_septic_shock":"category",
}
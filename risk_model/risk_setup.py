def load_data(data, outcome_ix=0):
    features, _, _, outcomes = get_risk_model_features()
    X = data[features].copy() # 患者特征
    y = data[outcomes[outcome_ix]] # 主要结局指标 28d死亡率
    return X,  y

def get_risk_model_vars():
    cate_vars = [var for var in risk_var_dict if risk_var_dict[var]=='category']
    cont_vars = [var for var in risk_var_dict if risk_var_dict[var]=='continuous']
    outcomes = ['28d_mortality', 'in_hospital_mortality']
    return cate_vars, cont_vars, outcomes

def get_risk_model_features():
    cate_vars, cont_vars, outcomes = get_risk_model_vars()
    cate_features = [var for var in cate_vars if not var in outcomes]
    cont_features = [var for var in cont_vars if not var in outcomes]
    features = [var for var in list(risk_var_dict.keys()) if not var in outcomes]
    return features, cate_features, cont_features, outcomes

risk_var_dict = {
"sex":"category",
"age":"continuous",
"BMI":"continuous",

"temperature":"continuous",
"heart_rate":"continuous",
"respir_rate":"continuous",
"SBP":"continuous",
"DBP":"continuous",
"MAP":"continuous",

"RBC":"continuous",
"WBC":"continuous",
"NE%":"continuous",
"LYM%":"continuous",
"Hb":"continuous",
# "HCT":"continuous",
"PLT":"continuous",

"ALT":"continuous",
"AST":"continuous",
"STB":"continuous",
"BUN":"continuous",
"Scr":"continuous",
"Glu":"continuous",
"K+":"continuous",
"Na+":"continuous",

"APTT":"continuous",
"Fg":"continuous",

"PH":"continuous",
"PaO2":"continuous",
"PaO2/FiO2":"continuous",
"PaCO2":"continuous",
"HCO3-":"continuous",
"Lac":"continuous",

"28d_mortality":"category",
"in_hospital_mortality":"category",
}
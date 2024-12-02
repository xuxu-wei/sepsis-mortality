
def load_data(data, dataset='EXIT_SEP'):
    features, _, _, outcomes = get_risk_model_features(dataset=dataset)
    X = data[features].copy() # 患者特征
    y = data[outcomes[0]] # 主要结局指标 28d死亡率
    return X,  y 

def get_risk_model_features(dataset='EXIT_SEP'):
    cate_vars, cont_vars, all_vars, outcomes = get_risk_model_vars(dataset=dataset)
    cate_features = [var for var in cate_vars if not var in outcomes]
    cont_features = [var for var in cont_vars if not var in outcomes]
    features = [var for var in all_vars if not var in outcomes]
    return features, cate_features, cont_features, outcomes

def get_risk_model_vars(dataset='EXIT_SEP'):
    
    if dataset.startswith('EXIT_SEP'):
        var_dict = {
        "XBJ_intervention":"category",
        "age":"continuous",
        "sex":"category",
        "BMI":"continuous",
        "primary_infection_site_lung":"category",
        "primary_infection_site_abdo":"category",
        "primary_infection_site_uri":"category",
        "primary_infection_site_skin":"category",
        "primary_infection_site_brain":"category",
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
        "SBP":"continuous",
        "DBP":"continuous",
        "SOFA_respiration":"category",
        "SOFA_coagulation":"category",
        "SOFA_liver":"category",
        "SOFA_cardio":"category",
        "SOFA_cns":"category",
        "SOFA_renal":"category",
        "SOFA":"category",
        "APACHE_II":"category",
        "DIC-score":"category",
        "septic_shock":"category",
        "MAP":"continuous",
        "Lac":"continuous",
        "Respiratory_Support":"category",
        "CCRT":"category",
        "nutri_support":"category",
        "nutri_support_enteral":"category",
        "nutri_support_parenteral":"category",
        "RBC":"continuous",
        "Hb":"continuous",
        "WBC":"continuous",
        "NE%":"continuous",
        "LYM%":"continuous",
        "PLT":"continuous",
        "HCT":"continuous",
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
        "D-Dimer":"continuous",
        "CRP":"continuous",
        "PCT":"continuous",
        "PH":"continuous",
        "PaO2/FiO2":"continuous",
        "HCO3-":"continuous",
        "PaO2":"continuous",
        "PaCO2":"continuous",
        "in_hospital_mortality":"category",
        "28d_mortality":"category",
        "7d_septic_shock":"category",
        }
        cate_vars = [var for var in var_dict if var_dict[var]=='category']
        cont_vars = [var for var in var_dict if var_dict[var]=='continuous']
        all_vars = [var for var in var_dict]
        outcomes = ['28d_mortality', '7d_septic_shock']

    else:
        raise ValueError(f'unexpected dataset label {dataset}')
    
    return cate_vars, cont_vars, all_vars, outcomes
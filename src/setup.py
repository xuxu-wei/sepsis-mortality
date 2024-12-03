import os, sys

src_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
# 将仓库路径添加到 sys.path
if src_path not in sys.path:
    sys.path.append(src_path)

# set data dir for local and server
if sys.platform.startswith('win'):
    SRC = src_path
    ROOT = f'{src_path}/../'
else:
    SRC = src_path
    ROOT = f'/home/xuxu.wei/sepsis-reasearch/'

DATA = f'{ROOT}/data/'
FIGS = f'{ROOT}/figures/'
TABLES = f'{ROOT}/tables/'
MODELS = f'{ROOT}/models/'

for workdir in [ROOT, DATA, FIGS, TABLES, MODELS]:
    if not os.path.exists(workdir):
        os.makedirs(workdir)
        print(f'work directory set up: {workdir}')

def get_cleaned_vars(dataset='EXIT_SEP'):
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
        "SOFA_respiration":"category", # PaO2/FiO2
        "SOFA_coagulation":"category", # PLT
        "SOFA_liver":"category", # 胆红素
        "SOFA_cardio":"category", # MAP 和 血管活性药物
        "SOFA_cns":"category", # GCS 评分
        "SOFA_renal":"category",# 血清肌酐和尿量
        "SOFA":"category",
        "APACHE_II":"category",
        "DIC-score":"category",
        "septic_shock":"category",
        "MAP":"continuous",
        "Lac":"continuous",
        "Respiratory_Support":"category",
        'MV':"category",
        'NIPPV':"category",
        'MV/NIPPV':"category",
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
        outcomes = ['28d_mortality', 'in_hospital_mortality', '7d_septic_shock']
        cate_vars = [var for var in var_dict if (var_dict[var]=='category') and ( not var in outcomes) ]
        cont_vars = [var for var in var_dict if (var_dict[var]=='continuous') and ( not var in outcomes) ]

    elif dataset.startswith('MIMIC_IV'):
        var_dict = {
        "age":'continuous',
        "sex":'category',
        "weight":'continuous',
        "height":'continuous',
        "BMI":'continuous',
        "temperature":'continuous',
        "heart_rate":'continuous',
        "respir_rate":'continuous',
        "SBP":'continuous',
        "DBP":'continuous',
        "MAP":'continuous',
        "SPO2":'continuous',
        "cancer":'category',
        # "liver_cirrhosis":'category',
        # "chronic_heart_failure":'category',
        "metastatic_cancer":'category',
        # "leukemia":'category',
        # "lymphoma":'category',
        # "myeloma":'category',
        "hematologic_cancer":'category',
        "AIDS":'category',
        
        "SOFA_respiration":'category',
        "SOFA_coagulation":'category',
        "SOFA_liver":'category',
        "SOFA_cardio":'category',
        "SOFA_cns":'category',
        "SOFA_renal":'category',
        "SOFA":'category',
        'GCS':'continuous',

        'APS_III':'continuous',
        'SIRS':'continuous',
        'SAPS_II':'continuous',
        'OASIS':'continuous',
        'Charlson':'continuous',

        "CCRT":'category',
        "MV":'category',
        "NIPPV":'category',
        "RBC":'continuous',
        "WBC":'continuous',
        "Hb":'continuous',
        "NE#":'continuous',
        "NE%":'continuous',
        "LYM#":'continuous',
        "LYM%":'continuous',
        "PLT":'continuous',
        "HCT":'continuous',
        "ALT":'continuous',
        "AST":'continuous',
        "STB":'continuous',
        "BUN":'continuous',
        "Scr":'continuous',
        # "HbA1c":'continuous',
        "Glu":'continuous',
        "K+":'continuous',
        "Na+":'continuous',
        "Ca2+":'continuous',
        # "Cl-":'continuous',
        "Fg":'continuous',
        "PT":'continuous',
        "APTT":'continuous',
        # "D-Dimer":'continuous',
        # "CRP":'continuous',
        "PH":'continuous',
        "PaO2/FiO2":'continuous',
        "HCO3-":'continuous',
        "PaO2":'continuous',
        "Lac":'continuous',
        "PaCO2":'continuous',
        "in_hospital_mortality":'category',
        "28d_mortality":'category',
        }
        outcomes = ['28d_mortality', 'in_hospital_mortality']
        cate_vars = [var for var in var_dict if (var_dict[var]=='category') and ( not var in outcomes) ]
        cont_vars = [var for var in var_dict if (var_dict[var]=='continuous') and ( not var in outcomes) ]

    else:
        raise ValueError(f'unexpected dataset label {dataset}')
    
    return cate_vars, cont_vars, outcomes
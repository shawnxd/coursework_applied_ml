training_data.csv
test.csv
Features and labels are contained in the file training_data.csv. 
You must predict labels for each of the instances in test.csv.

Here is the description of each column in the file.

patient_id: column 1 - the id of the patient

Features: Columns 2-29 are demographic features, columns 30-55 are responses to the questionnaire we've conducted on a sample population and columns 56 - 80 represents results from lab tests.

Labels: Columns 81, called 'label' indicates whether the surveyed person is infected by the disease. It is the outcome you will need to predict for the test dataset. 

-----------------------------------------------------------------------

Demographic features are:

'Release_cycle'     # virus data release cycle; 
'Status'            # examination status of the surveyed person; 1: only interviewed; 2: interviewed & MEC examined
'Gender'            # gender
'Age'               # age 
'Race1'             # Race/Hispanic origin; 1: Mexican American; 2: Other Hispanic 3: Non-Hispanic White 4: Non-Hispanic Black 5: Other race 
'Race2'             # Race/Hispanic origin w/ NH Asian; 1: Mexican American; 2: Other Hispanic 3: Non-Hispanic White 4: Non-Hispanic Black 5: Other race 
'Exam_period'       # the period the examination was performed; 1: examined in period 1 2: examined in period 2 
'Country'           # country born; 1: born in the U.S. 2: Other places 77: refuse to answer 99: not sure  
'US_citizen'        # U.S. citizen or not; 1: yes 2: no 7: refuse to answer 9: not sure 
'Language'          # spoken language; 1: English 2: Spanish 
'Self_respond_sp'   # whether a proxy respondent used in conducting the Sample Person interview; 1: yes 2: no 
'Self_respond2_sp'  # whether an interpreter used in conducting the Sample Person interview; 1: yes 2: no 
'Interview_lang'    # language used in the interview 1: English 2: Spanish 
'Self_respond_fam'  # whether a proxy respondent used in conducting the family interview; 1: yes 2: no 
'Self_respond2_fam' # whether an interpreter used in conducting the family interview; 1: yes 2: no 
'Num_household'     # total number of people in the Household
'Num_family'        # total number of people in the family 
'Num_young_child'   # number of children aged 5 years or younger in the household
'Num_adolescent'    # number of children aged 6-17 years old in the household
'Num_elder'         # number of elders aged 60 years or older in the household
'HH_gender'         # household reference person's gender 
'HH_age'            # household reference person's age 
'HH_country'        # household reference person's born country  
'HH_education'      # household reference person's education level 
'HH_marital'        # household reference person's marital status     
'Income_household'  # total household income 
'Income_family'     # total family income
'income_ratio'      # ratio of family income to poverty guidelines


Questions in the questionnaire are listed below. The participants are asked to what degree they agree with the statement. 
1: strongly agree  
2: agree 
3: disagree  
7: refuse to answer / not sure 

'Q1'                # I am constantly worried whether food would run out before I get money to buy more.
'Q2'                # The food that I bought lost its taste quickly.
'Q3'                # I can't afford to eat balanced meals.
'Q4'                # My family relies primarily on low-cost foods. 
'Q5'                # I couldn't feed my children a balanced meal. 
'Q6'                # My children are not eating enough because I can't afford it. 
'Q7'                # In the past couple months, my children are not eating enough. 
'Q8'                # In the past couple months, my children have skipped meals.  
'Q9'                # How often did the children skip meals due to budget constraint in the past couple months?
'Q10'               # In the past couple months, my children felt hungry multiple times 
'Q11'               # In the past couple months, my children had not eaten for an entire day. 
'Q12'               # In the past couple months, I have gotten emergency food from a church, a food pantry, or a food bank. 
'Q13'               # My children received care from a walk-in-clinic when he/she was 1 to 3 years old. 
'Q14'               # Adult food security category for last 12 months
'Q15'               # Children food security category for last 12 months
'Q16'               # Household food security category for last 12 months
'Q17'               # In the past couple months, I have received benefits from social programs. 
'Q18'               # I have previously received Food Stamp benefits. 
'Q19'               # In the past couple weeks, I have felt little interest or pleasure in doing things. 
'Q20'               # In the past couple weeks, I have felt down, depressed, or hopeless. 
'Q21'               # In the past couple weeks, I have trouble falling or staying asleep, or sleeping too much. 
'Q22'               # In the past couple weeks, I have had poor appetite or had been overeating. 
'Q23'               # In the past couple weeks, I have felt that I am a failure or have let myself or my family down?
'Q24'               # In the past couple weeks, I have had trouble concentrating on things, such as reading the newspaper or watching TV?
'Q25'               # In the past couple weeks, I have been bothered by speaking too slowly that other people could not notice?
'Q26'               # How difficult have these problems made it for me to do work, take care of things at home, or get along with people?

'Q27'               #In the past couple weeks, how often have you been bothered by the following problems: Thoughts that you would be better off dead or of hurting yourself in some way?

'Q28'                #In the past couple weeks, have you tried hurting yourself in some way?

'blood_pressure'       #Diastolic: Blood pres (1st rdg) mm Hg

'max_inflation_level'  #MIL: maximum inflation levels (mm Hg)

##Pulse regular or irregular?

'pulse_status' 
1 - regular pulse
2 - irregular pulse  

###Schema for the below variables is: 
##Description 
##Name 


##Average Sagittal Abdominal Diameter (cm)
'abdomen_diameter'

##BMI Category - Children/Adolescents
'weight_status' 
1 - underweight 
2 - normal weight 
3 - overweight 
4 - obese

Weight (kg)
'weight'

##Dentition Status Code
'dentition_status' 
1 - complete 
2 - partial 
3 - not done 

##Pregnancy test result
'pregnancy_test'
1 - positive 
2 - negative 
3 - not done 
4 - invalid 

##Fluoride, water (mg/L) average 2 values
'fluoride_water_level'

##Fluoride, plasma (umol/L) average 2 values
fluoride_plasma_level

##Urine #1 Flow Rate (mL/min)
urine_flow_rate

##A serious virus B with potential co-morbidity
virusB_infect_status
1 - positive 
2 - negative
3 - indeterminate 

##A serious virus C with potential co-morbidity
virusC_infect_status
1 - positive 
2 - negative
3 - indeterminate

##A serious virus D with potential co-morbidity
virusD_infect_status
1 - positive 
2 - negative
3 - indeterminate 

##Glycohemoglobin level % 
glycohemoglobin

##The time (in hours) the examinee last drank alcohol such as beer, wine, or liquor.
last_alcohol

##Have you had any of the following since "today at" or "yesterday at" the fasting time in ‘last_alcohol.
recent_alcohol
1 - yes 
2 - no 

##Hepatitis A antibody
hepatitis_A_test
1 - positive 
2 - negative 
3 - indeterminate

##Hepatitis D (anti-HDV)
hepatitis_D_test
1 - positive 
2 - negative 
3 - indeterminate 

##Hepatitis B core antibody (Anti-HBc)
hepatitis_B_test
1 - positive 
2 - negative 
3 - indeterminate 

##Total calcium (mmol/L)
calcium_level 

##Glucose, serum (mmol/L)
glucose_level 

##Iron level 
iron_level 

##A serious virus E with potential co-morbidity
virusE_infect_status 
1 - positive 
2 - negative
3 - indeterminate 
-----------------------------------------------------------------------
-----------------------------------------------------------------------
-----------------------------------------------------------------------












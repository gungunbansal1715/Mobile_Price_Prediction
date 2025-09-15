import streamlit as st
import pandas as pd
import pickle

st.title("PHONE PRICE PREDICTION APP")
tab1 , tab2 = st.tabs(["HOME" , "ABOUT US"])

with tab1:
    battery_power = st.number_input("Enter battery power" , min_value = 100 , max_value = 2000)
    blue = st.selectbox("Is bluethooth available?(0-no or 1-yes)" , [0,1])
    clock_speed = st.number_input("Enter the clock speed" , min_value = 0.0 , max_value = 5.0 , value = 0.0)
    dual_sim = st.selectbox("Does your phone contains dual sim or not? 0 - No and 1 - Yes" , [0,1])
    fc= st.number_input("Enter the pixels of your front camera" , min_value = 0, max_value = 20 , value = 0 )
    four_g = st.selectbox("Does your phone has 4g connection? 0:No and 1:Yes" , [0,1])
    int_memory = st.number_input("Enter your internal memory", min_value = 0 , max_value = 100)
    m_dep = st.number_input("Enter your phone's depth" , min_value = 0.0 , max_value = 1.0 , value = 0.0)
    m_weight = st.number_input("Enter your phone's weight" , min_value = 50 , max_value = 200)
    n_cores = st.number_input("Enter number of CPU cores", min_value = 1 , max_value = 10)
    pc = st.number_input("Enter your phone's primary camera pixels" , min_value = 0 , max_value = 100)
    px_height = st.number_input("Enter your phone's pixels resolution height" , min_value = 0, max_value = 5000)
    px_width = st.number_input("Enter your phone's pixels resolution width", min_value = 0 , max_value = 5000)
    ram = st.number_input("Enter RAM of your phone", min_value = 100, max_value = 5000)
    sc_h = st.number_input("Enter the screen height" , min_value=0 , max_value=50)
    sc_w = st.number_input("Enter screen width", min_value= 0 , max_value=50)
    talk_time = st.number_input("Enter the battery talk time", min_value=0 , max_value=20)
    three_g = st.selectbox("Does your phone supports 3g connectivity? 0 - No or 1 - yes", [0,1])
    touch_screen = st.selectbox("Does your phone has touch screen or not?0 - no , 1 - yes", [0,1])
    wifi = st.selectbox("Does your phone wifi? 0- No or 1- yes", [0,1])

    input_data = pd.DataFrame({
        'battery_power' : [battery_power],
        'blue' : [blue],
        'clock_speed' : [clock_speed],
        'dual_sim' : [dual_sim],
        'fc' : [fc],
        'four_g' : [four_g],
        'int_memory' : [int_memory],
        'm_dep' : [m_dep],
        'mobile_wt' : [m_weight],
        'n_cores'  : [n_cores],
        'pc' : [pc],
        'px_height' : [px_height],
        'px_width' : [px_width],
        'ram' : [ram],
        'sc_h' : [sc_h],
        'sc_w' : [sc_w],
        'talk_time' : [talk_time],
        'three_g' : [three_g],
        'touch_screen' : [touch_screen],
        'wifi' : [wifi],
        })
    algo_names = ['Logistic Regression' , 'Decision Tree' , 'Random Forest' , 'SVM']
    model_names = ['LR.pkl' , 'DTree.pkl' , 'SVM.pkl', 'RandomForest.pkl']
    selected_algo = st.selectbox("Select prediction algorithm", algo_names)
    model_file = model_names[algo_names.index(selected_algo)]

    if st.button("PREDICT"):
        with open(model_file , 'rb') as f:
            model = pickle.load(f)
        predicted_price = model.predict(input_data)[0]
        st.success(f"Estimated price range(0-low price, 1- medium low price, 2- medium high price , 3 - high price): {int(predicted_price):}")
    
        
with tab2:
    st.title("About this project")
    st.info("""
        This project focuses on building a machine learning based system to predict the pricing category of mobile phones using their technical specifications and features.\n
        The primary goal of this project is to classify the price of phone in 4 ccategories:\n
            0 - Low price\n
            1 - Medium low price\n
            2 - Medium high price\n
            3 - High price\n
        The approach demonstrate the practical application of MACHINE LEARNING in market analysis and product pricing.\n
        Output of this project may or may not be correct as it is developed at a learning stage by using the pre-given dataset and that data can be modified further accordingly...\n
            
        The project is developed by:\n
            GUNGUN BANSAL\n
            contact: bansalgungun203@gmail.com\n
            A data science enthusiast\n
         This project is developed as a part of machine learning course on Unified Mentor.
""")
    import plotly.express as px
    data = {
        'Logistic Regression' : 0.97,
        'Decision Tree' : 0.83,
        'Random Forest': 0.83,
        'SVM' : 0.98
    }
    models = list(data.keys())
    accuracy = list(data.values())
    df = pd.DataFrame(list(zip(models , accuracy)), columns = ['model' , 'accuracy'])
    fig = px.pie(df , values = 'accuracy', names='model' , title='MOdel Accuracy Comparison')
    st.plotly_chart(fig)
    fig1 = px.bar(df, x='model' , y='accuracy', title='Model Accuracy Comparison', text='accuracy')
    st.plotly_chart(fig1)
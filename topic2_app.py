import streamlit as st
import pandas as pd
import numpy as np
import random
import re
import regex
import time
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import warnings
warnings.filterwarnings('ignore')
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from underthesea import word_tokenize, sent_tokenize, pos_tag
from sklearn.pipeline import Pipeline
import pickle
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from deep_translator import GoogleTranslator
from gensim import corpora, models, similarities
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split  
from sklearn. metrics import classification_report
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, precision_score
# import gensim
import scipy
import mylibs as ml
# # ĐỌC FILE
@st.cache_data
def load_df_company(file):
    return pd.read_excel(file, engine="openpyxl",index_col='id')

@st.cache_data
def load_df_reviews(file):
    return pd.read_excel(file, engine="openpyxl",index_col='id')

df_Company = load_df_company('Overview_Companies.xlsx')
df_Reviews = load_df_reviews('Reviews.xlsx')

data = df_Reviews.merge(df_Company, on='Company Name', how='left')
data.columns = data.columns.str.replace("’","'")


# Cấu hình trang
st.set_page_config(layout="wide")
# CSS để đẩy footer xuống đáy sidebar
st.markdown("""
    <style>
    /* Tăng chiều cao sidebar và đẩy footer xuống đáy */
    [data-testid="stSidebar"] > div:first-child {
        display: flex;
        flex-direction: column;
        height: 100vh; /* full chiều cao trình duyệt */
    }

    .sidebar-footer {
        margin-top: auto;
        font-size: 12px;
        color: gray;
        padding-top : 50px;
        line-height: 1.5;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar (menu bên trái)
# Khởi tạo trạng thái nếu chưa có
if "selected_button" not in st.session_state:
    st.session_state.selected_button = "1"  # mặc định hiển thị nội dung 1

# Sidebar: Các nút
with st.sidebar:
    st.title("📂 Menu")

    if st.button("Overview"):
        st.session_state.selected_button = "1"
    if st.button("Content-Based"):
        st.session_state.selected_button = "2"
    if st.button("Recommender System"):
        st.session_state.selected_button = "3"
    if st.button("New Prediction"):
        st.session_state.selected_button = "4"
    # Footer nằm ở đáy sidebar
    st.markdown("""
        <div class='sidebar-footer'>
            <hr style="margin-top: 10px; margin-bottom: 5px;">
            Các thành viên thực hiện</br><br>
            👤1. Phạm Nhật Minh<br>
            📧 <a href="mailto:mphamm12@gmail.com">mphamm12@gmail.com</a><br><br>
            👤2. Võ Quốc Hùng<br>
            📧 <a href="mailto:hung232803@gmail.com">hung232803@gmail.com</a>      
            
        </div>
        
    """, unsafe_allow_html=True)
    # st.write("✅ Gensim version:", gensim.__version__)
# Tạo 2 cột
col1, col2 = st.columns([4, 1])  # tỷ lệ 5:1 cho nội dung và logo

with col1:
    st.markdown("""
        <h1 style='text-align: center; color: #2c3e50;'>
            ĐỒ ÁN TỐT NGHIỆP DATA SCIENCE - MACHINE LEARNING - K304
        </h1>
        <h3 style='text-align: center; color: #2c3e50;'>
            Giáo viên hướng dẫn: Khuất Thùy Phương
        </h3>

    """, unsafe_allow_html=True)

with col2:
    st.image("logo_nhom.png",width=200)

# Đường kẻ ngăn cách
st.markdown("---")

# Hiển thị nội dung theo mục đã chọn
if st.session_state.selected_button == "1":
    ml.scroll_to_top()
    st.markdown("""
        <h3 style='text-align: center; color: #2c3e50;'>
            📊 Có 2 chủ đề trong khóa học:
        </h3>
    """, unsafe_allow_html=True)
    # Tạo 2 cột
    col1, col2 = st.columns([5, 5])  # tỷ lệ 5:1 cho nội dung và logo
    with col1:
        st.markdown("""
        <h5 style='text-align: left; color: #2c3e50;'>
            Topic 1: Content-Based Filtering
        </h5>
    """, unsafe_allow_html=True)        
        st.image("Clustering_p1.png", width=300, caption="Sentiment Analysis & Clustering")
    with col2:
        st.markdown("""
        <h5 style='text-align: center; color: #2c3e50;'>
            Topic 2: Recommender System
        </h5>
    """, unsafe_allow_html=True)
        st.image("recommend.png", width=300, caption="Recommender System") 
        
#  CONTENTBASED
if st.session_state.selected_button == "2":
    ml.scroll_to_top()
    
    st.markdown("""
    # 📈 Gensim
    - [Gensim trên PyPI](https://pypi.org/project/gensim): Là một thư viện Python chuyên xác định sự tương tự về ngữ nghĩa giữa hai tài liệu thông qua mô hình không gian vector và bộ công cụ mô hình hóa chủ đề.
    - Có thể xử lý kho dữ liệu văn bản lớn với sự trợ giúp của việc truyền dữ liệu hiệu quả và các thuật toán tăng cường
    - Tốc độ xử lý và tối ưu hóa việc sử dụng bộ nhớ tốt
    - Tuy nhiên, Gensim có ít tùy chọn tùy biến cho các function

    #### 🔗 Tham khảo thêm:
    - [TutorialsPoint - Gensim](https://www.tutorialspoint.com/gensim/index.htm)
    - [MachineLearningPlus - Gensim Tutorial](https://www.machinelearningplus.com/nlp/gensim-tutorial)
    """)
    # Open and read file to cosine_sim_new
    with open('companies_cosine_sim.pkl', 'rb') as f:
        cosine_sim_new = pickle.load(f)    

    st.image('img/channels4_banner.jpg')
    
    if 'random_companies' not in st.session_state:
        df_companies = pd.read_excel('Overview_Companies.xlsx')
        st.session_state.random_companies = df_companies.sample(n=10, random_state=42)
    else:
        df_companies = pd.read_excel('Overview_Companies.xlsx')

    

    # Kiểm tra xem 'selected_id' đã có trong session_state hay chưa
    if 'selected_id' not in st.session_state:
        # Nếu chưa có, thiết lập giá trị mặc định là None hoặc ID sản phẩm đầu tiên
        st.session_state.selected_id = None
    # Kiểm tra xem 'selected_id' đã có trong session_state hay chưa
    if 'selected_id' not in st.session_state:
        # Nếu chưa có, thiết lập giá trị mặc định là None hoặc ID sản phẩm đầu tiên
        st.session_state.selected_id = None
    
    # Theo cách cho người dùng chọn công ty từ dropdown
    # Tạo một tuple cho mỗi sản phẩm, trong đó phần tử đầu là tên và phần tử thứ hai là ID
    company_options = [(row['Company Name'], row['id']) for index, row in st.session_state.random_companies.iterrows()]
    # st.session_state.random_companies
    # Tạo một dropdown với options là các tuple này
    selected_company = st.selectbox(
        "Chọn công ty",
        options=company_options,
        format_func=lambda x: x[0]  # Hiển thị tên công ty
    )
    # Display the selected company
    st.write("Bạn đã chọn:", selected_company)
    # Cập nhật session_state dựa trên lựa chọn hiện tại
    st.session_state.selected_id = selected_company[1]
    
    if st.session_state.selected_id:
        st.write("id: ", st.session_state.selected_id)
        # Hiển thị thông tin sản phẩm được chọn
        selected_company = df_companies[df_companies['id'] == st.session_state.selected_id]
        if not selected_company.empty:
            # st.write('#### Bạn vừa chọn:')
            st.write('### ', selected_company['Company Name'].values[0])

            company_description = selected_company['Company overview'].values[0]
            truncated_description = ' '.join(company_description.split()[:100])
            st.write('##### Information:')
            st.write(truncated_description, '...')

            st.write('##### Các công ty liên quan:')
            recommendations = ml.get_recommendations(df_companies, st.session_state.selected_id, cosine_sim=cosine_sim_new, nums=3) 
            ml.display_recommended_companies_row(recommendations)
        else:
            st.write(f"Không tìm thấy công ty với ID: {st.session_state.selected_id}")

# RECOMMEND
if st.session_state.selected_button == "3":
    ml.scroll_to_top()
    st.subheader("🤖 Tạo Module dự đoán xem có Recommender hay là không?")
    st.subheader("Một số công ty có mặt trên thị trường")
    # st.dataframe(data)
   # Gom nhóm theo Company Name, nối các Title và What I like
    grouped_data = data.groupby('Company Name').agg({
        'Title': lambda x: '\n'.join(x.dropna().astype(str)),
        'What I liked': lambda x: '\n'.join(x.dropna().astype(str)) ,
        'Rating': 'mean'  
    }).reset_index()
    # Bước 2: Chọn ngẫu nhiên 5 công ty
    sampled = grouped_data.sample(n=5)  # có thể bỏ random_state để luôn random khác
    if st.button("🎲 Chọn 5 công ty ngẫu nhiên khác"):
        sampled = sampled.sample(n=5)
    def truncate_text(text, max_length=100):
        return text if len(text) <= max_length else text[:max_length] + "..."    

    # Hiển thị từng công ty
    for _, row in sampled.iterrows():
        preview_title = truncate_text(row['Title'], 100)
        preview_like = truncate_text(row['What I liked'], 100)

        with st.expander(f"🏢 {row['Company Name']} — ⭐ Trung bình rating: {row['Rating']:.2f}"):
            st.markdown(f"**🎯 Titles:** {preview_title}")
            st.markdown(f"**👍 What they liked:** {preview_like}")

            with st.expander("📖 Xem chi tiết"):
                st.markdown(f"**🎯 Full Titles:**\n{row['Title']}")
                st.markdown(f"**👍 Full What they liked:**\n{row['What I liked']}")

        
    st.subheader("II. Chart")
    
    # Vẽ biểu đồ phân bố sl người
    st.markdown("<h4 style='margin-left: 20px;'>🔹Qui mô nhân sự</h4>", unsafe_allow_html=True) 
           
    st.image("img/df_nhansu.png")
    st.image('img/c1_nhansu.png')

    # Biểu đồ quốc gia:
    st.markdown("<h4 style='margin-left: 20px;'>🔹Mật độ phân bố các quốc gia</h4>", unsafe_allow_html=True)        
    
    st.image("img/df_quocgia.png")
    st.image("img/c2_quocgia.png")
    
    # Thời gian làm việc
    st.image("img/df_thoigian.png")
    st.image("img/c3_thoigian.png")

    # OT
    st.markdown("<h4 style='margin-left: 20px;'>🔹Có OT không?</h4>", unsafe_allow_html=True)        

    st.image("img/df_ot.png")
    st.image("img/c4_ot.png")
    
    st.subheader("III. Built Model")

    # Áp dụng style và hiển thị
    st.image("img/df_model_I.png")
    st.image("img/c5_model_tot.png")
    
    st.markdown("---")
    
    st.image("img/c6_confusion.png")

    st.image("img/c7_roc.png")
    
if st.session_state.selected_button == "4":
    ml.scroll_to_top()
    st.subheader("New Prediction")
    
    df_companies = data
    if 'random_companies' not in st.session_state:       
        st.session_state.random_companies = df_companies.sample(n=10, random_state=42)

    if 'selected_id' not in st.session_state:
        st.session_state.selected_id = None
    company_options = [(row['Company Name'], index) for index, row in st.session_state.random_companies.iterrows()]

    selected_company = st.selectbox(
        "Chọn công ty",
        options=company_options,
        format_func=lambda x: x[0]
    )
    # Display the selected company
    st.write("""
        <svg width="24" height="24" fill="none" stroke="red" stroke-width="3" stroke-linecap="round" viewBox="0 0 24 24">
        <path d="M12 5v14M5 12h14"/>
        </svg> <span style="font-size:18px;">Bạn đã chọn công ty: </span>
        """, selected_company[0],unsafe_allow_html=True)    
    st.write("🔑 ID:", selected_company[1])
    st.markdown(
        """
        <hr style="border-top: 1.5px dashed green; width: 100%;">
        """,
        unsafe_allow_html=True
        )
    # Cập nhật session_state dựa trên lựa chọn hiện tại
    st.session_state.selected_id = selected_company[1]
    # DÙNG CHO CÁC SELECTION DROPDOWNS
    selections = {}  # Dùng dict để lưu kết quả từng dropdown
    dropdowns = [
        ("Company size", sorted(data['Company size'].dropna().unique(), key=ml.extract_min)),
        ("Country", data['Country'].dropna().unique()),
        ("Working days", data['Working days'].dropna().unique()),
        ("Overtime Policy", data['Overtime Policy'].dropna().unique())
        ]
    # Tạo từng dòng form có label + selectbox canh ngang
    for label, options in dropdowns:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown(f"<p style='margin-top: 0.6rem'>{label}</p>", unsafe_allow_html=True)
        with col2:
            selected = st.selectbox(label="",
                                options=options,
                                label_visibility="collapsed")
            selections[label] = selected  # Gán giá trị vào dict
    # DÙNG CHO CÁC SLIDER
    sliders = [
        ("Rating", 1, 5, 4),   # (label, min, max, default)
        ("Salary & benefits", 1, 5, 4),
        ("Training & learning", 1, 5, 4),
        ("Management cares about me", 1, 5, 4),
        ("Culture & fun", 1, 5, 4),
        ("Office & workspace", 1, 5, 4)
        ]

    slider_results = {}

    for i, (label, min_val, max_val, default_val) in enumerate(sliders):
        col1, col2,col3  = st.columns([1,1,1])
        with col1:
            st.markdown(f"<p style='margin-top: 0.6rem'>{label}</p>", unsafe_allow_html=True)
        with col2:
            value = st.slider(
                label="",
                min_value=min_val,
                max_value=max_val,
                value=default_val,
                key=f"slider_{i}"
            )
            slider_results[label] = value

    # Sử dụng điều khiển submit
    # st.subheader("Recommender")
    submitted = st.button("Submit")
    if submitted:
        st.write("Bạn đã chọn:")
        col1, col2 = st.columns([1,1])
        with col1:
            for label, value in selections.items():
                st.write(f"✅ **{label}**: {value}") 
        with col2:
            for label, val in slider_results.items():
                st.write(f"🔹 {label}: {val}")        

        # Tạo DataFrame từ dict (1 dòng)
        input_df = pd.DataFrame([{**selections, **slider_results}])
        
        # Load model và encoders
        with open("logistic_model.pkl", "rb") as f:
            recommended = pickle.load(f)
        
        for col in input_df.columns:
            if input_df[col].dtype == 'object':
                le = LabelEncoder()
                input_df[col] = le.fit_transform(input_df[col].astype(str))
        
        predit_new = recommended.predict(input_df)
        result_map = {1: 'Recommend', 0: 'No Recommend'}
        recommend_label = result_map[predit_new[0]]

        st.success(f"✅ Kết quả dự đoán Recommend?: {recommend_label}")
        
        # st.success(f"🔮 Kết quả dự đoán: {predit_new[0]}")
       




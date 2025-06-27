import streamlit as st
import pandas as pd
import numpy as np
import random
import re
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import warnings
warnings.filterwarnings('ignore')
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
import pickle
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split  
from sklearn. metrics import classification_report
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, precision_score

# ĐỌC FILE
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

for col in ['Company industry','Working days', 'Overtime Policy','Our key skills']:
    mode_val = data[col].mode()
    if  not mode_val.empty:
        data[col] = data[col].fillna(mode_val[0])

data.dropna(subset=['What I liked','Suggestions for improvement'],inplace=True)

X = data[['Company size', 'Country','Working days','Overtime Policy',
'Rating', 'Salary & benefits','Training & learning','Management cares about me',
'Culture & fun','Office & workspace']].copy()
data['Recommend?'] = np.where(data['Recommend?'] == 'Yes',1,0 )
y = data['Recommend?']


label_encoders = {}  # Dictionary để lưu encoder theo tên cột
for col in X.columns:
    if X[col].dtype == 'object':
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[f"le_{col}"] = le  # Đặt tên encoder theo dạng 'le_<column>'
        
        
X_train, X_test, y_train, y_test =train_test_split(X, y, test_size= 0.3)
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'DecisionTree':DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}
results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # cần cho ROC AUC

    results.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'ROC AUC': roc_auc_score(y_test, y_proba)
    })
# Định nghĩa hàm tô màu dòng
def highlight_row(row):
    if row['Model'] == 'Logistic Regression':  # đổi điều kiện nếu muốn
        return ['background-color: #ffcccc'] * len(row)
    else:
        return [''] * len(row)
def scroll_to_top():
    st.markdown("""
        <script>
            window.scrollTo(0, 0);
        </script>
    """, unsafe_allow_html=True)
    
# LƯU MODEL
logistic_model = models['Logistic Regression']
pkl_filename = "logistic_model.pkl"  
# Lưu
with open(pkl_filename, 'wb') as f:  
    pickle.dump({
        'model': logistic_model,
        'label_encoders': label_encoders
    }, f)

# Load lại
with open(pkl_filename, 'rb') as f:
    saved = pickle.load(f)

model = saved['model']
label_encoders = saved['label_encoders'] 
    
# Hàm lấy số nhỏ nhất trong chuỗi, ví dụ '51-200' → 51
def extract_min(company_size):
    if pd.isna(company_size):
        return float('inf')  # Đẩy giá trị NaN xuống cuối
    match = re.findall(r'\d+', company_size)
    return int(match[0]) if match else float('inf')


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

    if st.button("🧭 Overview"):
        st.session_state.selected_button = "1"
    if st.button("🧩 1.Content-Based"):
        st.session_state.selected_button = "2"
    if st.button("🤖 2.Recommender System"):
        st.session_state.selected_button = "3"
    if st.button("📈 New Prediction"):
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

    
# Tạo 2 cột
col1, col2 = st.columns([4, 1])  # tỷ lệ 5:1 cho nội dung và logo

with col1:
    st.markdown("""
        <h1 style='text-align: center; color: #2c3e50;'>
            ĐỒ ÁN TỐT NGHIỆP DATA SCIENCE - MACHINE LEARNING - K304
        </h1>
        <h3 style='text-align: center; color: #2c3e50;'>
            Giáo viên hướng dẫn: Khuất Thụy Phương
        </h3>

    """, unsafe_allow_html=True)

with col2:
    st.image("logo_nhom.png",width=200)

# Đường kẻ ngăn cách
st.markdown("---")

# Hiển thị nội dung theo mục đã chọn
if st.session_state.selected_button == "1":
    scroll_to_top()
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
        
    
if st.session_state.selected_button == "2":
    scroll_to_top()
    st.subheader("📈 Phân tích chi tiết")
    st.write("Bao gồm phân tích theo hành vi, nhóm tuổi, doanh thu...")

if st.session_state.selected_button == "3":
    scroll_to_top()
    st.subheader("🤖 Tạo Module dự đoán xem có Recommender hay là không?")
    st.subheader("I. Xem nguồn dữ liệu")
    st.write(data['Company Name'].value_counts().head(5))
    st.subheader("II. Chart")
    
    # Vẽ biểu đồ phân bố sl người
    st.markdown("<h4 style='margin-left: 20px;'>🔹Qui mô nhân sự</h4>", unsafe_allow_html=True)        
    df_employees = df_Company.value_counts('Company size').reset_index()
    df_employees.columns = ['Company size','count']
    st.write(df_employees.head(5))
    fig = px.pie(df_employees, values='count', names='Company size',title='Biểu đồ Phân bố qui mô số lượng người')
    fig.update_traces(text=df_employees['Company size'], textposition='outside')
    fig.update_layout(margin=dict(t=60, l=0, r=0, b=80),
                      title_font_size=20,
                      title_font_color='Red',
                      height = 500)
    st.plotly_chart(fig, use_container_width=True)

    # Biểu đồ quốc gia:
    st.markdown("<h4 style='margin-left: 20px;'>🔹Mật độ phân bố các quốc gia</h4>", unsafe_allow_html=True)        
    df_country = df_Company.value_counts('Country').reset_index()
    df_country.columns = ['Country','count']
    top_10_country = df_country.head(10)
    st.write(top_10_country.head(5))
    fig = px.pie(top_10_country, values='count', names='Country',title='Biểu đồ Phân bố các quốc gia') #hole= 0.2)
    fig.update_traces(text=top_10_country['Country'], textposition='outside')
    fig.update_layout(margin=dict(t=60, l=0, r=0, b=80),
                      title_font_size=24,height=500,
                      title_font_color='Red')    
    st.plotly_chart(fig, use_container_width=True)
    
    # Thời gian làm việc
    st.markdown("<h4 style='margin-left: 20px;'>🔹Thời gian làm việc</h4>", unsafe_allow_html=True)        
    df_working = df_Company.value_counts('Working days').reset_index()
    df_working.columns = ['Working days','count']
    st.write(df_working.head(5))
    fig = px.pie(df_working, values='count', names='Working days',title='Biểu đồ thời gian làm việc')
    fig.update_traces(text=df_working['Working days'], textposition='outside',rotation=90,)
    fig.update_layout(margin=dict(t=80, l=0, r=0, b=80),height=500,
                      title_font_size=24, title_font_color='Red')
    st.plotly_chart(fig, use_container_width=True)
    
    # OT
    st.markdown("<h4 style='margin-left: 20px;'>🔹Có OT không?</h4>", unsafe_allow_html=True)        
    df_ot = df_Company.value_counts('Overtime Policy').reset_index()
    df_ot.columns = ['Overtime Policy','count']
    st.write(df_ot)
    fig = px.pie(df_ot, values='count', names='Overtime Policy',title='Biểu đồ thể hiện có làm việc OT không')
    fig.update_traces(text=df_ot['Overtime Policy'], textposition='outside',rotation=90,)
    fig.update_layout(margin=dict(t=60, l=0, r=0, b=80),height=500,
                      title_font_size=24, title_font_color='Red')
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("III. Built Model")
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(by='F1-Score', ascending=False)
    # st.write(df_results)
    # Áp dụng style và hiển thị
    styled_df = df_results.style.apply(highlight_row, axis=1)
    st.dataframe(styled_df, use_container_width=True)

    ax = plt.figure(figsize=(10, 6))
    sns.set(font_scale=0.8)
    sns.heatmap(df_results.set_index('Model'), annot=True, fmt=".3f", cmap="viridis")
    plt.title("Model Performance Comparison")
    st.pyplot(ax)
    
    st.markdown("---")
    
    # Danh sách màu sắc cmap tương ứng (đủ 5 màu đẹp và khác nhau)
    colormaps = ['Blues', 'Oranges', 'Purples', 'Greens']
    ax1 = plt.figure(figsize=(10, 10))
    for i, (name, model) in enumerate(models.items()):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        # Vẽ từng confusion matrix
        plt.subplot(2, 2, i + 1)
        sns.heatmap(cm, annot=True, fmt='d', cmap=colormaps[i], cbar=False)
        plt.title(f"Confusion Matrix - {name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")

    plt.tight_layout()
    st.pyplot(ax1)
    
    ax2 = plt.figure(figsize=(10, 8))

    for name, model in models.items():
        model.fit(X_train, y_train)

        # Lấy xác suất dự đoán lớp 1
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        else:
            y_score = model.decision_function(X_test)

        # Tính FPR, TPR, AUC
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.2f})')

    # Đường chéo random model
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend(loc='lower right')
    plt.grid(True)
    st.pyplot(ax2)
    
if st.session_state.selected_button == "4":
    scroll_to_top()
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
        ("Company size", sorted(data['Company size'].dropna().unique(), key=extract_min)),
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

        # st.write("🎯 Dữ liệu đầu vào dạng bảng:")
        # st.dataframe(input_df)
        
        saved = pickle.load(open("logistic_model.pkl", "rb"))
        model = saved['model']
        label_encoders = saved['label_encoders']      
        for col in input_df.columns:
            key = f'le_{col}'
            if key in label_encoders:
                input_df[col] = label_encoders[key].transform(input_df[col].astype(str))

        # if input_df.select_dtypes(include='object').shape[1] > 0:
        #     st.error("⛔ Vẫn còn cột kiểu object chưa được mã hóa!")
        #     # Lấy các cột có kiểu dữ liệu object
        #     object_cols = input_df.select_dtypes(include='object').columns.tolist()

        #     # Hiển thị ra
        #     st.warning(f"⚠️ Còn các cột chưa được mã hóa: {object_cols}")
        # else:
            # prediction = model.predict(input_df)
            # st.success(f"🔮 Kết quả dự đoán: {prediction[0]}")
        predit_new = model.predict(input_df)
        st.success(f"🔮 Kết quả dự đoán: {predit_new[0]}")




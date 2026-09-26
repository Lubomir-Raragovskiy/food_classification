import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

MODEL_PATH = 'food_classifier.keras' 
IMAGE_SIZE = (128, 128)  
class_names = ['apple_pie', 'falafel', 'french_toast', 'ice_cream', 'ramen', 'sushi', 'tiramisu']

st.set_page_config(page_title="CNN Розпізнавання зображень", page_icon="🖼️", layout="centered")

st.title("Класифікація зображень за допомогою CNN")
st.write("Завантажте одне або декілька зображень, і нейронна мережа спробує визначити, що на них.")

with st.expander("Доступні класи для розпізнавання", expanded=True):
    st.markdown("""
    **Поточна версія моделі натренована розпізнавати 7 категорій страв:**
    * 🥧 **Яблучний пиріг** (Apple pie)
    * 🧆 **Фалафель** (Falafel)
    * 🍞 **Французькі тости** (French toast)
    * 🍨 **Морозиво** (Ice cream)
    * 🍜 **Рамен** (Ramen)
    * 🍣 **Суші** (Sushi)
    * 🍰 **Тірамісу** (Tiramisu)
    
    *Для найкращих результатів завантажуйте чіткі фотографії, де страва знаходиться в центрі кадру.*
    """)

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

try:
    model = load_model()
    st.success("Модель успішно завантажена!")
except Exception as e:
    st.error(f"Помилка завантаження моделі: Перевірте файл {MODEL_PATH}. Деталі: {e}")
    st.stop()

st.divider()

st.subheader("Завантаження зображень")
uploaded_files = st.file_uploader("Оберіть файли зображень", type=["jpg", "jpeg", "png", "bmp", "webp"], accept_multiple_files=True)

if uploaded_files:
    st.info(f"Завантажено зображень: {len(uploaded_files)}")
    
    if st.button("Розпізнати всі зображення", use_container_width=True):
        st.divider()
        st.subheader("Результати розпізнавання:")
        
        for idx, uploaded_file in enumerate(uploaded_files):
            st.markdown(f"### Зображення {idx + 1}: {uploaded_file.name}")
            
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.image(image, caption='Завантажене зображення', use_container_width=True)
            
            with col2:
                with st.spinner("Аналіз..."):

                    if image.mode != "RGB":
                        image = image.convert("RGB")
                        
                    img_resized = image.resize(IMAGE_SIZE)
                    img_array = np.array(img_resized)
                    img_array = img_array / 255.0 
                    img_array = np.expand_dims(img_array, axis=0) 
                    
    
                    try:
                        predictions = model.predict(img_array)
                        
                        predicted_class_index = np.argmax(predictions[0])
                        confidence = np.max(predictions[0]) * 100
                        
                        if predicted_class_index < len(class_names):
                            predicted_label = class_names[predicted_class_index]
                        else:
                            predicted_label = f"Клас #{predicted_class_index}"
                        
                        st.success(f"**Знайдено об'єкт:** {predicted_label}")
                        st.info(f"**Впевненість моделі:** {confidence:.2f}%")
                        
                        with st.expander("Детальні ймовірності"):
                            for i, prob in enumerate(predictions[0]):
                                label = class_names[i] if i < len(class_names) else f"Клас {i}"
                                st.write(f"{label}: {prob*100:.1f}%")
                                st.progress(float(prob))
                                
                    except Exception as e:
                        st.error(f"Помилка під час прогнозування: {e}")
            
            st.divider()
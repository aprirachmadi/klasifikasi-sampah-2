import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import tempfile
from streamlit_option_menu import option_menu

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('./model/model.h5')
    return model

model = load_model()

# Function to preprocess image
def preprocess_image(image):
    image = image.resize((128, 128))
    image = np.array(image)
    if image.shape[-1] == 4:
        image = image[:, :, :3]
    image = np.expand_dims(image, axis=0)
    return image

output_dict = {
    0 : 'Anorganik',
    1 : 'Organik'
}

# Define the navigation bar
page = option_menu(
    menu_title=None,
    options=['Home', 'Menu Utama', 'Tentang Kami'],
    orientation='horizontal'
)

# Home Page
if page == "Home":
    st.title("pilahIn")
    st.markdown("""
    ### Aplikasi untuk Memilah Sampah Organik dan Anorganik
    """)
    st.markdown("""
    ### Selamat Datang di Aplikasi pilahIn
    Sebuah aplikasi yang dapat membantu pengguna untuk dapat MEMILAH sampah organik dan sampah anorganik
    Hanya dalam 1 kali klik pada aplikasi
    """)
    st.markdown("""
    ### Langkah Penggunaan
    Buka "Menu Utama" -> unggah atau ambil gambar -> klik pada tombol "prediksi"
    """)

# Main Menu Page
elif page == "Menu Utama":
    st.title("Klasifikasi Sampah Organik dan anorganik")
    st.write("Unggah file atau ambil foto")

    # File uploader
    uploaded_file = st.file_uploader("Pilih file", type=["jpg", "jpeg", "png"])

    # Take a picture
    picture = st.camera_input("Ambil foto")

    def preprocess_image(image):
        image = image.resize((128, 128))
        image = np.array(image)
        if image.shape[-1] == 4:
            image = image[:, :, :3]
        image = np.expand_dims(image, axis=0)
        return image
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
        image = Image.open(uploaded_file)

    if picture is not None:
        st.image(picture, caption='Taken Picture', use_column_width=True)
        image = Image.open(picture)

    # Initialize the history
    if 'history' not in st.session_state:
        st.session_state.history = []

    button = st.button("Prediksi")

    if button:
        st.write("Classifying...")
    
        image = preprocess_image(image)
        preds = model.predict(image)
    
        pred_class = (preds > 0.5).astype(int)[0]

        label = 'Organik' if preds > 0.5 else 'Anorganik'
        st.write(f"Prediksi: {label} ")

        # Record history
        st.session_state.history.append((label))

        if pred_class == 1:
            st.write("""
        ## Apa itu sampah Organik?
        Sampah organik adalah jenis sampah yang berasal dari sisa-sisa makhluk hidup, baik hewan, tanaman, maupun manusia. Sampah organik dapat terurai secara alami di lingkungan.
        ## Jenis-jenis sampah Organik.
        Sampah organik dapat dibagi menjadi dua jenis berdasarkan kandungan airnya: basah dan kering.
        1. Sampah Organik Basah
        Contoh: sisa sayuran, buah-buahan, kulit pisang, kulit bawang, dan sejenisnya.
        Ciri: banyak mengandung air, sehingga lebih cepat membusuk dan terurai.
        2. Sampah Organik Kering
        Contoh: ranting pohon, dedaunan kering, tulang belulang, dan sejenisnya.
        Ciri: kandungan air sedikit hingga tidak ada, sehingga membutuhkan waktu lebih lama untuk terurai.
        ## Pemanfaatan sampah Organik.
        Sampah organik dapat diolah menjadi bahan yang berguna melalui beberapa cara, seperti mengompos, pembuatan pelet makanan untuk ayam dan ikan, dan penggunaan sebagai bahan bakar untuk menghasilkan listrik melalui proses biogas. 
        Dengan demikian, pengelolaan sampah organik yang tepat dapat mengurangi volume sampah, menghasilkan kompos yang berguna, dan mengurangi bau tidak sedap serta gas metana.
        """)
        if pred_class == 0:
            st.write("""
        ## Apa itu sampah Anorganik?
        Sampah anorganik adalah produk buangan atau zat sisa yang sulit atau tidak dapat diurai secara alami. Contoh sampah anorganik meliputi plastik, logam, kaca, kertas, dan keramik
        ## Pemanfaatan sampah Anorganik
        1. Daur Ulang
        Sampah anorganik seperti plastik, kertas, dan kaca dapat didaur ulang menjadi produk baru. Contoh: botol plastik dapat diubah menjadi pot bunga atau tempat pensil.
        2. Bank Sampah 
        Sampah anorganik dapat dijual ke bank sampah untuk didaur ulang. Contoh: plastik, kertas, dan logam dapat dijual ke bank sampah untuk diproses menjadi bahan baku yang dapat digunakan kembali.
        3. Pemanfaatan Kembali
        Sampah anorganik seperti botol plastik dapat dimanfaatkan kembali sebagai tempat penyimpanan atau dekorasi rumah
         """)
        
    st.write("### Riwayat Prediksi")
    for i, (label) in enumerate(st.session_state.history):
        st.write(f"{i + 1}. {label}")

# About Us Page
elif page == "Tentang Kami":
    st.title("Tentang Kami")
    st.write("""
    ### pilahIn
    Aplikasi ini dikembangkan oleh Tim 2 KKN Universitas Diponegoro Desa Banyakprodo 2024
    """)
    st.image('logo sekolah.jpg')
    st.write("""
    pilahIn berdedikasi untuk memberikan edukasi pemilahan sampah sejak dini di seluruh lapisan masyarakat, membangun Indonesia yang lebih hijau dan ramah lingkungan melalui teknologi yang inovatif dan mudah diakses.
    """)

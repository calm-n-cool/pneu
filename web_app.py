import tensorflow as tf
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing import image
from PIL import Image
from keras.preprocessing import image
from tempfile import NamedTemporaryFile

st.set_option('deprecation.showfileUploaderEncoding', False)

classifier = tf.keras.models.load_model("classifier.h5")

STYLE = """
<style>
img {

    max-width: 50%;
}
</style>
"""

pneumonia_html="""  
      <div style="background-color:#F08080;padding:10px >
       <h2 style="color:black ;text-align:center;"> Prediction: Pneumatic Lung</h2>
       </div>
    """
    
    
normal_html="""  
      <div style="background-color:#74F61D;padding:10px >
       <h2 style="color:white;text-align:center;"> Prediction: Normal Lung </h2>
       </div>
    """


st.title("Lung X-RAY classification")
st.markdown(STYLE, unsafe_allow_html = True)

file = st.file_uploader("Upload the Lung X-RAY image to be analysed", type= ["PNG", "JPEG","JPG"])

show_file = st.empty()
temp_file = NamedTemporaryFile(delete=False)

if not file:
    show_file.info("Please upload a file of type: " + ", ".join(["PNG","JPEG"]))
else: 
    show_image = Image.open(file)  
    #test_image = test_image.resize((64,64))
    
    temp_file.write(file.getvalue())
    test_image = image.load_img(temp_file.name, target_size = (64, 64))
    
    fig = plt.figure()
    plt.imshow(show_image)
    plt.axis("off")
    st.pyplot(fig)
    
    test_image = image.img_to_array(test_image)
    
    #test_image = np.array(test_image)
    #test_image = test_image[:,:,0]
    
    test_image = np.expand_dims(test_image,axis=0)
    result = classifier.predict(test_image)
    if (result[0][0]) == 1:
        st.markdown(pneumonia_html,unsafe_allow_html=True)
    else:
        st.markdown(normal_html,unsafe_allow_html=True)


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import streamlit as st
from keras.preprocessing import image
from keras import backend as K
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
from tempfile import NamedTemporaryFile
from util import load_C3M3_model

# Disable eager execution
tf.compat.v1.disable_eager_execution()
tf.get_logger().setLevel('ERROR')
np.random.seed(0)

# Load the model
model_path = "./"  # Update with the actual path
model = load_C3M3_model(model_path)

# Load labels
labels = ['Cardiomegaly', 'Emphysema', 'Effusion', 'Hernia', 'Infiltration', 'Mass', 'Nodule', 'Atelectasis',
          'Pneumothorax', 'Pleural_Thickening', 'Pneumonia', 'Fibrosis', 'Edema', 'Consolidation']

# Load CSV data
data_dir = 'data/nih_new/images-small/'  # Update with the actual path
df = pd.read_csv('data/nih_new/train-small.csv')

# Helper function to compute Grad-CAM
def grad_cam(input_model, image, category_index, layer_name):

    cam = None

    output_with_batch_dim = input_model.output
    output_all_categories = output_with_batch_dim[0]
    y_c = output_all_categories[category_index]
    spatial_map_layer = input_model.get_layer(layer_name).output
    grads_l = K.gradients(y_c, spatial_map_layer)
    grads = grads_l[0]
    spatial_map_and_gradient_function = K.function([input_model.input], [spatial_map_layer, grads])
    spatial_map_all_dims, grads_val_all_dims = spatial_map_and_gradient_function([image])
    spatial_map_val = spatial_map_all_dims[0]
    grads_val = grads_val_all_dims[0]
    weights = np.mean(grads_val, axis = (0, 1))
    cam = np.dot(spatial_map_val, weights)

    H, W = image.shape[1], image.shape[2]
    cam = np.maximum(cam, 0) # ReLU so we only get positive importance
    cam = cv2.resize(cam, (W, H), cv2.INTER_NEAREST)
    cam = cam / cam.max()

    return cam

# Helper function to load and preprocess image
def get_mean_std_per_batch(IMAGE_DIR, df, H=320, W=320):
    sample_data = []
    for idx, img in enumerate(df.sample(100)["Image"].values):
        path = IMAGE_DIR + img
        sample_data.append(np.array(image.load_img(path, target_size=(H, W))))

    mean = np.mean(sample_data[0])
    std = np.std(sample_data[0])
    return mean, std    

def load_image_normalize(path, mean, std, H=320, W=320):
    x = image.img_to_array(image.load_img(path, target_size=(H, W)))
    x -= mean
    x /= std
    x = np.expand_dims(x, axis=0)
    return x

def load_image(path, df, preprocess=True, H=320, W=320):
    """Load and preprocess image."""
    x = image.img_to_array(image.load_img(path, target_size=(H, W)))
    if preprocess:
        mean, std = get_mean_std_per_batch(data_dir, df, H=H, W=W)
        x -= mean
        x /= std
        x = np.expand_dims(x, axis=0)
    return x

# Streamlit GUI
def main():
    st.title("Grad-CAM Visualization")

    # Display the "Choose File" button
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    # Create a layout with two columns
    col1, col2 = st.columns([1, 1])
    # Display the "Start" button in the second column
    with col2:
        start_prediction = st.button("Start Prediction")

    if uploaded_file is not None:
        # Save the uploaded file as a temporary file
        with NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.read())
            temp_path = temp_file.name

        # Preprocess the uploaded image
        img = image.load_img(temp_path, target_size=(320, 320))
        img_array = image.img_to_array(img)

        mean, std = get_mean_std_per_batch(data_dir, df)
        preprocessed_input = load_image_normalize(temp_path, mean, std)

        # Display the input and output images side by side
        col_input, col_output = st.columns(2)
        with col_input:
            st.image(img, caption='Input Image', use_column_width=True)
        with col_output:
            if start_prediction:
                # Predict using the model
                predictions = model.predict(preprocessed_input)
                highest_prob_label_index = np.argmax(predictions)
                highest_prob_label = labels[highest_prob_label_index]

                # Compute and display Grad-CAM
                gradcam = grad_cam(model, preprocessed_input, highest_prob_label_index, 'conv5_block16_concat')
                gradcam = cv2.resize(gradcam, (320, 320), cv2.INTER_LINEAR)
                # Apply the heatmap as an overlay on the original image
                heatmap = cv2.applyColorMap(np.uint8(255 * gradcam), cv2.COLORMAP_JET)[..., ::-1]
                heatmap = heatmap.astype(np.float32) / 255.0

                # Apply a mask to limit the heatmap to the detected regions
                heatmap[heatmap < 0.2] = 0  # Set low values to 0 to ignore them
                heatmap = heatmap / np.max(heatmap)  # Normalize heatmap values

                # Blend the heatmap with the original image
                masked_img = (img_array / 255.0) * 0.8 + (heatmap * 0.4)  # Adjust alpha and beta values as needed

                # Ensure pixel values are within [0.0, 1.0] range
                masked_img = np.clip(masked_img, 0.0, 1.0)

                # Display the output image and text
                st.image(masked_img, caption='Grad-CAM Heatmap', use_column_width=True)
                st.write(f"Diagnosis: {highest_prob_label} with a percentage of {round(predictions[0][highest_prob_label_index] * 100, 2)}%")


if __name__ == "__main__":
    main()
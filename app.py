import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from models.emotion_resnet18 import EmotionResNet18


# Emotion class labels
classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
@st.cache_resource
def load_model():
    model = EmotionResNet18().to(device)
    model.load_state_dict(torch.load("improved_emotion_resnet18.pth", map_location=device))
    model.eval()
    return model

# Prediction function (now uses PIL image directly)
def predict_emotion(pil_image, model):
    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
                         ])

    image = transform(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        probs = torch.nn.functional.softmax(output, dim=1).squeeze()

        # Top 3 predictions
        top3_probs, top3_indices = torch.topk(probs, 3)
        predictions = [(classes[top3_indices[i]], top3_probs[i].item() * 100) for i in range(3)]
        return predictions

# Streamlit UI
st.set_page_config(page_title="Emotion Detection App", layout="centered")
st.title("😊 Emotion Detection App")
st.write("Upload a face image to detect the top 3 emotions.")

# Load model
try:
    model = load_model()
except FileNotFoundError:
    st.error("❌ Model file 'improved_emotion_resnet18.pth' not found.")
    model = None

# Upload image and show predictions
if model:
    uploaded_file = st.file_uploader("Upload an image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_container_width=True)

            # Predict
            predictions = predict_emotion(image, model)

            st.subheader("🔍 Top Emotion Predictions:")
            for emotion, probability in predictions:
                st.write(f"**{emotion}**: {probability:.2f}%")

            # Final result
            st.success(f"✅ Most likely emotion: **{predictions[0][0]}**")

        except Exception as e:
            st.error(f"⚠️ Error processing image: {e}")

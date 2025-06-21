import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models

# Nombre de clases
class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

# Descripciones breves para cada flor
flower_descriptions = {
    'daisy': "🌼 **Daisy**: Pequeña y encantadora, la margarita simboliza la pureza y la inocencia. Requiere mucha luz y riego moderado.",
    'dandelion': "🌼 **Dandelion**: Conocido por sus semillas voladoras, el diente de león es resistente y tiene propiedades medicinales.",
    'rose': "🌹 **Rose**: Una de las flores más populares, representa el amor. Existen muchas variedades y colores.",
    'sunflower': "🌻 **Sunflower**: Gira hacia el sol y es símbolo de alegría y energía. Necesita mucho sol y espacio para crecer.",
    'tulip': "🌷 **Tulip**: Elegante y de floración primaveral, es símbolo de renovación. Prefiere climas frescos y suelos bien drenados."
}

# Transformaciones para la imagen
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Cargar modelo
@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(torch.load("modelo_flores.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# Interfaz
st.title("🌸 Clasificador de Flores")
st.write("Sube una imagen de flor y el modelo te dirá cuál es junto con una breve descripción.")

uploaded_file = st.file_uploader("📷 Sube una imagen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Imagen subida', use_column_width=True)

    # Preprocesar imagen
    img_tensor = transform(image).unsqueeze(0)  # batch dimension

    # Predicción
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
        pred_class = class_names[predicted.item()]

    st.success(f"La flor predicha es: **{pred_class.upper()}**")

    # Mostrar descripción
    st.info(flower_descriptions[pred_class])

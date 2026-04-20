import streamlit as st
from PIL import Image
import numpy as np
import sys
import os
import nibabel as nib
from io import BytesIO

try:
    import pydicom
except ImportError:
    pydicom = None

# Adicionar o diretório raiz ao path para imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference import load_model, predict_mask, predict_prob_map, predict_prob_volume, predict_report
from src.utils import overlay_mask, plot_comparison, create_heatmap, dice_coefficient, iou_score, sensitivity_score, specificity_score, f1_score

st.set_page_config(page_title="MedVision Assist", layout="wide")

st.title("MedVision Assist")
st.markdown("Aplicação médica para segmentação e apoio à decisão baseada em imagem.")

uploaded_file = st.file_uploader(
    "Envie uma imagem médica para inferência",
    type=["nii", "nii.gz", "png", "jpg", "jpeg", "dcm"],
)
uploaded_mask = st.file_uploader(
    "Envie uma máscara de referência (opcional)",
    type=["nii", "nii.gz", "png", "jpg", "jpeg", "dcm"],
)

if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    filename = uploaded_file.name.lower()

    nifti_volume = None
    slice_index = None
    run_volume_inference = False
    if filename.endswith(".nii") or filename.endswith(".nii.gz"):
        nifti_bytes = BytesIO(file_bytes)
        nifti = nib.load(nifti_bytes)
        image_array = nifti.get_fdata()
        if image_array.ndim == 4:
            image_volume = image_array[:, :, :, 0]
        else:
            image_volume = image_array

        if image_volume.ndim != 3:
            st.error("Formato NIfTI não suportado: espere um volume 3D ou 4D.")
            st.stop()

        nifti_volume = image_volume
        depth = nifti_volume.shape[2]
        slice_index = st.sidebar.slider("Selecione o slice", 0, depth - 1, depth // 2)
        run_volume_inference = st.sidebar.checkbox("Inferência volumétrica completa", value=False)
        image_slice = nifti_volume[:, :, slice_index]
        image_slice = ((image_slice - np.nanmin(image_slice)) / (np.nanmax(image_slice) - np.nanmin(image_slice)) * 255).astype(np.uint8)
        image = Image.fromarray(image_slice).convert("RGB")
    elif filename.endswith('.dcm'):
        if pydicom is None:
            st.error('Instale pydicom para processar DICOM.')
            st.stop()
        dicom = pydicom.dcmread(BytesIO(file_bytes))
        image_array = dicom.pixel_array
        image_array = ((image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array)) * 255).astype(np.uint8)
        image = Image.fromarray(image_array).convert("RGB")
    else:
        image = Image.open(BytesIO(file_bytes)).convert("RGB")

    st.sidebar.image(image, caption="Imagem carregada", use_column_width=True)
    if nifti_volume is not None:
        st.sidebar.write(f"Volume carregado: {nifti_volume.shape}")
        st.sidebar.write(f"Slice selecionado: {slice_index}")
        st.sidebar.write(f"Inferência volumétrica: {'ativada' if run_volume_inference else 'desativada'}")

    st.header("Preview da imagem")
    st.image(image, use_column_width=True)

    with st.spinner("Carregando modelo e gerando resultados..."):
        try:
            model = load_model()
        except FileNotFoundError as exc:
            st.error(str(exc))
            st.stop()

        image_np = np.array(image)
        if run_volume_inference and nifti_volume is not None:
            prob_volume = predict_prob_volume(model, nifti_volume)
            prob_map = prob_volume[:, :, slice_index]
            mask = (prob_map > 0.5).astype(np.uint8)
        else:
            mask = predict_mask(model, image_np)
            prob_map = predict_prob_map(model, image_np)

        report = predict_report(model, image_np)
        heatmap = create_heatmap(image_np, prob_map)

    st.header("Segmentação")
    st.image(overlay_mask(image_np, mask), caption="Segmentação sobreposta", use_column_width=True)

    st.header("Heatmap de probabilidade")
    st.image(heatmap, caption="Heatmap sobreposto", use_column_width=True)

    if uploaded_mask is not None:
        mask_bytes = uploaded_mask.read()
        mask_name = uploaded_mask.name.lower()
        try:
            if mask_name.endswith(".nii") or mask_name.endswith(".nii.gz"):
                mask_nii = nib.load(BytesIO(mask_bytes))
                mask_array = mask_nii.get_fdata()
                if mask_array.ndim == 4:
                    mask_array = mask_array[:, :, :, 0]
                if mask_array.ndim == 3 and slice_index is not None:
                    mask_array = mask_array[:, :, slice_index]
                mask_array = (mask_array > 0).astype(np.uint8)
            elif mask_name.endswith('.dcm'):
                if pydicom is None:
                    raise RuntimeError('pydicom não está instalado para processar máscaras DICOM')
                mask_dicom = pydicom.dcmread(BytesIO(mask_bytes))
                mask_array = (mask_dicom.pixel_array > 0).astype(np.uint8)
            else:
                mask_image = Image.open(BytesIO(mask_bytes)).convert("L")
                mask_array = np.array(mask_image.resize(image.size)).astype(np.uint8) > 127

            clinical_metrics = {
                'dice': dice_coefficient(mask, mask_array),
                'iou': iou_score(mask, mask_array),
                'sensitivity': sensitivity_score(mask, mask_array),
                'specificity': specificity_score(mask, mask_array),
                'f1': f1_score(mask, mask_array),
            }
            st.subheader("Métricas de referência")
            st.json(clinical_metrics)
        except Exception as error:
            st.warning(f"Não foi possível processar a máscara de referência: {error}")

    st.header("Relatório de apoio")
    st.write(report)

    st.header("Comparação")
    fig = plot_comparison(image_np, mask)
    st.pyplot(fig)

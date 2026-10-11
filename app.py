from pathlib import Path
from collections import Counter
import csv
import hashlib
import io
import logging
import os
import shutil
import tempfile
import zipfile

import streamlit as st
import timm
import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from torchvision import transforms

ROOT = Path(__file__).resolve().parent
CLASSES = ['Coccidiosis', 'Healthy', 'New Castle Disease', 'Salmonella']
MODEL = 'convnextv2_tiny.fcmae_ft_in22k_in1k'
WEIGHTS = Path(os.getenv('CURRICULUM_WEIGHTS', str(ROOT / 'weights/best_curriculum.pth')))
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32 if DEVICE.type == 'cuda' else 8
MAX_IMAGES = 10000
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_TOTAL_BYTES = 5 * 1024 * 1024 * 1024

st.set_page_config(page_title='Poultry · Curriculum Classifier', page_icon='🐔', layout='wide')
st.markdown('''<style>
.stApp {background:#0e1117;color:#f1f5f9;color-scheme:dark;}
[data-testid="stHeader"] {background:#0e1117;}
.stApp h1,.stApp h2,.stApp h3,.stApp p,.stApp label {color:#f1f5f9;}
[data-testid="stCaptionContainer"] p {color:#a9b8cb;}
[data-testid="stFileUploaderDropzone"] {background:#1b2330;color:#f1f5f9;border:1px solid #354256;}
[data-testid="stFileUploaderDropzone"] button {background:#263247;color:#f1f5f9;border-color:#52657e;}
.block-container {max-width:1120px;padding-top:2.5rem;}
h1 {letter-spacing:-.04em;}
[data-testid="stVerticalBlockBorderWrapper"] {background:#161e2b;border-color:#354256;border-radius:18px;}
.stButton>button {border-radius:10px;}
</style>''', unsafe_allow_html=True)


@st.cache_resource(max_entries=1, show_spinner=False)
def load_model(path: str, modified_ns: int, size: int, device_name: str):
    # Only load a trusted checkpoint installed on the server, never user uploads.
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError('Expected a state dictionary or checkpoint dictionary.')
    state = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
    state = {k.removeprefix('module.'): v for k, v in state.items()}
    names = checkpoint.get('class_names')
    if names is None and isinstance(checkpoint.get('experiment_signature'), dict):
        names = checkpoint['experiment_signature'].get('class_names')
    names = list(names or CLASSES)
    if len(names) != 4 or set(names) != set(CLASSES):
        raise ValueError('Checkpoint must contain the four expected poultry classes.')

    model = timm.create_model(MODEL, pretrained=False, num_classes=4)
    model.load_state_dict(state, strict=True)
    model.to(torch.device(device_name)).eval()
    config = timm.data.resolve_model_data_config(model)
    transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(config['mean'], config['std']),
    ])
    return model, names, transform


def get_model():
    info = WEIGHTS.stat()
    return load_model(str(WEIGHTS), info.st_mtime_ns, info.st_size, str(DEVICE))


def open_image(data: bytes):
    with Image.open(io.BytesIO(data)) as source:
        image = ImageOps.exif_transpose(source)
        image.load()
        return image.convert('RGB')


def classify(image, model, names, transform):
    tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        index = model(tensor).argmax(dim=1).item()
    return names[index]


def true_label_from_path(filename):
    # Ground truth comes from a class-named folder, never from the model output.
    # Example: test/Healthy/image.jpg -> Healthy
    parent_parts = Path(filename.replace('\\', '/')).parts[:-1]
    def normalized(value):
        return ''.join(char for char in value.casefold() if char.isalnum())
    class_lookup = {normalized(name): name for name in CLASSES}
    for folder in reversed(parent_parts):
        if normalized(folder) in class_lookup:
            return class_lookup[normalized(folder)]
    return 'Unknown'


def process_zip(uploaded_file, model, names, transform):
    results = []
    counts = Counter()
    progress = st.progress(0)
    status = st.empty()
    temporary_path = None
    try:
        # Back the ZIP with a temporary disk file instead of calling getvalue().
        uploaded_file.seek(0)
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp:
            temporary_path = temp.name
            shutil.copyfileobj(uploaded_file, temp)

        with zipfile.ZipFile(temporary_path) as archive:
            members = [
                member for member in archive.infolist()
                if not member.is_dir()
                and not member.filename.startswith('__MACOSX/')
                and not Path(member.filename).name.startswith('.')
                and Path(member.filename).suffix.lower() in IMAGE_EXTENSIONS
            ]
            if not members:
                raise ValueError('No supported JPG, PNG, or WebP images were found in the ZIP.')
            if len(members) > MAX_IMAGES:
                raise ValueError(f'The ZIP exceeds the {MAX_IMAGES:,}-image limit.')
            if sum(member.file_size for member in members) > MAX_TOTAL_BYTES:
                raise ValueError('The uncompressed images exceed the 5 GB limit.')

            total = len(members)
            for start in range(0, total, BATCH_SIZE):
                batch_members = members[start:start + BATCH_SIZE]
                tensors, filenames = [], []
                for member in batch_members:
                    try:
                        if member.file_size > MAX_IMAGE_BYTES:
                            raise ValueError('Image is larger than 20 MB.')
                        with archive.open(member) as source:
                            image = open_image(source.read())
                        tensors.append(transform(image))
                        filenames.append(member.filename)
                    except (OSError, ValueError, UnidentifiedImageError,
                            Image.DecompressionBombError, RuntimeError):
                        results.append({'Filename': member.filename, 'True Label': true_label_from_path(member.filename), 'Prediction': 'Failed'})

                if tensors:
                    inputs = torch.stack(tensors).to(DEVICE)
                    with torch.inference_mode():
                        indices = model(inputs).argmax(dim=1).cpu().tolist()
                    for filename, index in zip(filenames, indices):
                        prediction = names[index]
                        counts[prediction] += 1
                        results.append({'Filename': filename, 'True Label': true_label_from_path(filename), 'Prediction': prediction})
                    del inputs

                completed = min(start + BATCH_SIZE, total)
                progress.progress(completed / total)
                status.caption(f'Classified {completed:,} / {total:,} images on {DEVICE.type.upper()}')
    finally:
        progress.empty()
        status.empty()
        if temporary_path is not None:
            try:
                os.unlink(temporary_path)
            except OSError:
                pass
    return results, counts


@st.dialog('Image prediction', width='large')
def show_prediction_image(uploaded_file, selected_result):
    filename = selected_result['Filename']
    st.write(f"**Filename:** {filename}")
    st.write(f"**True label:** {selected_result['True Label']}")
    st.write(f"**Prediction:** {selected_result['Prediction']}")
    try:
        uploaded_file.seek(0)
        with zipfile.ZipFile(uploaded_file, 'r') as archive:
            member = archive.getinfo(filename)
            if member.file_size > MAX_IMAGE_BYTES:
                raise ValueError('Image exceeds the 20 MB preview limit.')
            with archive.open(member) as source:
                image = open_image(source.read())
        st.image(image, caption=filename, width='stretch')
    except (KeyError, OSError, ValueError, zipfile.BadZipFile,
            Image.DecompressionBombError, UnidentifiedImageError):
        st.error('Unable to preview this image from the ZIP.')


def results_csv(results):
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=['Filename', 'True Label', 'Prediction'])
    writer.writeheader()
    writer.writerows(results)
    return stream.getvalue().encode('utf-8-sig')


st.caption('POULTRY FECAL CLASSIFICATION')
st.title('ConvNeXtV2 · Curriculum Classifier')
st.write('Classify a single poultry fecal image or an entire ZIP using the Curriculum-trained model.')
if DEVICE.type == 'cuda':
    st.caption(f'Inference: GPU · {torch.cuda.get_device_name(0)} · batch size {BATCH_SIZE}')
else:
    st.caption('Inference: CPU (CUDA not available)')

if not WEIGHTS.is_file():
    st.error(f'Curriculum checkpoint not found: {WEIGHTS}')
    st.stop()

single_tab, zip_tab = st.tabs(['Single image', 'ZIP batch classification'])

with single_tab:
    uploaded_image = st.file_uploader('Upload a chicken fecal image', type=['jpg', 'jpeg', 'png', 'webp'], key='single')
    if uploaded_image is not None:
        try:
            if uploaded_image.size > MAX_IMAGE_BYTES:
                raise ValueError('Image exceeds 20 MB.')
            image = open_image(uploaded_image.getvalue())
        except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError):
            st.error('This image could not be opened. Upload a valid image under 20 MB.')
        else:
            image_col, result_col = st.columns([1, 1.35], gap='large')
            with image_col:
                st.image(image, caption=uploaded_image.name, width='stretch')
            with result_col:
                with st.container(border=True):
                    st.subheader('Curriculum model')
                    if st.button('Classify image', type='primary', width='stretch'):
                        try:
                            with st.spinner('Classifying…'):
                                model, names, transform = get_model()
                                prediction = classify(image, model, names, transform)
                            st.caption('PREDICTED CLASSIFICATION')
                            st.markdown(f'### {prediction}')
                        except Exception:
                            logging.exception('Single-image classification failed')
                            st.error('Classification failed. Check server logs.')
                    else:
                        st.write('Ready to classify')

with zip_tab:
    st.write('Upload a ZIP containing class-named folders (e.g. Healthy/image.jpg). The folder provides the true label; predictions are generated by the model. Images without a recognized class folder show Unknown.')
    uploaded_zip = st.file_uploader('Upload ZIP file', type=['zip'], key='batch')
    if uploaded_zip is not None:
        # Reset results if the selected upload changes, even when the filename is the same.
        upload_id = (uploaded_zip.name, uploaded_zip.size, hashlib.sha256(uploaded_zip.getvalue()).hexdigest())
        if st.session_state.get('batch_upload_id') != upload_id:
            st.session_state['batch_upload_id'] = upload_id
            st.session_state.pop('batch_results', None)
            st.session_state.pop('batch_counts', None)
            st.session_state.pop('prediction_table', None)
        if st.button('Classify ZIP images', type='primary', width='stretch'):
            try:
                with st.spinner('Loading Curriculum model…'):
                    model, names, transform = get_model()
                results, counts = process_zip(uploaded_zip, model, names, transform)
                st.session_state['batch_results'] = results
                st.session_state['batch_counts'] = counts
            except (ValueError, zipfile.BadZipFile, OSError) as exc:
                st.error(f'Unable to process ZIP: {exc}')
            except Exception:
                logging.exception('ZIP classification failed')
                st.error('Batch classification failed. Check server logs.')

        if 'batch_results' in st.session_state:
            results = st.session_state['batch_results']
            counts = st.session_state['batch_counts']
            successful = sum(counts.values())
            st.divider()
            st.subheader('Batch results')
            a, b, c = st.columns(3)
            a.metric('Total images', len(results))
            b.metric('Classified', successful)
            c.metric('Failed', len(results) - successful)
            st.subheader('Predicted class distribution')
            for col, name in zip(st.columns(4), CLASSES):
                col.metric(name, counts.get(name, 0))
            st.subheader('Individual predictions')
            st.caption('Select an image row, then click Preview selected image.')
            selection = st.dataframe(
                results,
                width='stretch',
                hide_index=True,
                on_select='rerun',
                selection_mode='single-row',
                key='prediction_table',
            )
            selected_rows = selection.selection.rows
            if st.button(
                'Preview selected image',
                disabled=not bool(selected_rows),
                key='open_batch_preview',
            ):
                show_prediction_image(uploaded_zip, results[selected_rows[0]])
            st.download_button(
                'Download predictions CSV',
                data=results_csv(results),
                file_name='curriculum_predictions.csv',
                mime='text/csv',
                type='primary',
                width='stretch',
            )

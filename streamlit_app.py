import numpy as np
import streamlit as st
from insightface.app import FaceAnalysis
from PIL import Image

from utils import calc_cosine_similarity, get_emb

IMAGE_HEIGHT_DEST = 250
MODEL_NAME = "buffalo_l"
THRESHOLD = 0.36786744

face_model = FaceAnalysis(
    name=MODEL_NAME, root="./", allowed_modules=["detection", "recognition"]
)
face_model.prepare(ctx_id=-1, det_size=(640, 640))

st.set_page_config(page_title="Face Verification", page_icon="🧐", layout="wide")
st.write(
    "<style>{}</style>".format(open("./templates/style.css").read()),
    unsafe_allow_html=True,
)
st.write(
    '<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">',
    unsafe_allow_html=True,
)

st.write(
    open("./templates/nav.html").read(),
    unsafe_allow_html=True,
)

st.title("Face Verification Application")
st.write(
    "#### How does the computer know 2 pictures are the same person?",
    unsafe_allow_html=True,
)


def load_image(image_file, element_display=None):
    img = Image.open(image_file).convert("RGB")
    img = np.array(img)

    img_height, img_width = img.shape[:2]
    img_width_dest = int(img_width * IMAGE_HEIGHT_DEST / img_height)

    if element_display:
        element_display.image(img, width=img_width_dest)

    return img


def verify():
    if file_1 is not None and file_2 is not None:
        img_1 = load_image(file_1)
        img_2 = load_image(file_2)

        emb_1 = get_emb(img_1, face_model)
        emb_2 = get_emb(img_2, face_model)

        cos_sim = calc_cosine_similarity(emb_1, emb_2)

        center_block_result.json(
            {
                "similarity": cos_sim,
                "threshold": THRESHOLD,
                "method": "cosine_similarity",
            }
        )
        if cos_sim > THRESHOLD:
            center_block_result.info("They are the same person")

        else:
            center_block_result.warning("They are two different people")

    else:
        center_block_result.error("Images are invalid")


_, img_1, img_2, _ = st.columns(4)
_, col_1, col_2, _ = st.columns(4)
file_1 = col_1.file_uploader("Picture 1", type=["png", "jpg"])
file_2 = col_2.file_uploader("Picture 2", type=["png", "jpg"])

if file_1 is not None:
    load_image(file_1, img_1)

if file_2 is not None:
    load_image(file_2, img_2)

_, center_block_result, _ = st.columns((1, 2, 1))
center_block_result.button("Apply", on_click=verify)

_, center_block, _ = st.columns((1, 2, 1))
center_block.write("Example Image")
col_1, col_2, col_3, col_4, col_5, col_6 = st.columns(6)
load_image("./data/test_data/neymar_1.jpg", col_1)
load_image("./data/test_data/neymar_2.jpg", col_2)
load_image("./data/test_data/neymar_3.jpg", col_3)
load_image("./data/test_data/messi_1.jpg", col_4)
load_image("./data/test_data/messi_2.jpg", col_5)
load_image("./data/test_data/messi_3.jpg", col_6)

_, center_block, _ = st.columns((1, 6, 1))
center_block.header("Project Overview")
center_block.write(
    "You are the person, and you can determine that 2 pictures are the same person, or 2 different people. So how can the machine have that ability, in the time of digization."
)
center_block.write(
    "This application applys the archivement in Computer Vision, using Convolutional Neural Network and Cosine Similarity to verify 2 picture."
)
center_block.write(
    "The application of this project can use to many fields, like eKYC in bank, school, ... to save manpower, energy and more security"
)
center_block.write("""Technical Step:""")
center_block.write(
    """
        - Face Detection
        - Face Alignment
        - Feature Extraction
        - Cosine Similarity
"""
)
center_block.write("""Model:""")
center_block.write(
    """
        - RetinaFace for Face Detection
        - ArcFace for Face Extraction
"""
)
center_block.header("Data Analysis")
center_block.write(
    "The dataset using in this project is [Labeled Faces In The Wild (LFW)](http://vis-www.cs.umass.edu/lfw/), an offical benchmark dataset in many face verification and face recognition fields. LFW is collected, labeled and publiced in  IEEE Computer Vision and Pattern Recognition 2007 by University of Massachusetts, Amherst"
)
center_block.code(
    """
@TechReport{LFWTech,
    author =       {Gary B. Huang and Manu Ramesh and Tamara Berg and 
                    Erik Learned-Miller},
    title =        {Labeled Faces in the Wild: A Database for Studying 
                    Face Recognition in Unconstrained Environments},
    institution =  {University of Massachusetts, Amherst},
    year =         2007,
    number =       {07-49},
    month =        {October}
}"""
)
center_block.write(
    "LFW includes **13233** images, **5749** classes, and **1680** classes have more than 2 images, image size 250x250 pixel. In a image could have many faces."
)
center_block.write(
    "Faces in LFW is in many conditions, like positions, poses, brighness, hue, image quality, ages, genders, ... "
)
center_block.image(load_image("./img/lfw_example.png"))
center_block.header("Methodology")
center_block.write("Data Processing:")
center_block.write(
    """
    - Get all faces in an image, include the bounding box, landmark of face by RetinaFace model
    - Select the most central face in the image, using the position of bounding box of that face
    - Get the embedding of the most central face
    - Normalization the embedding by L2 Normalization
"""
)

center_block.write("Statistics for find the acceptance threshold")
center_block.write(
    """
    With a person in 1680 people, I will:
    - Get randomly 2 images of him, named img_1 and img_2
    - Calculate embedding of 2 image, named emb_1 and emb_2, respectively
    - Append into array emb_A and emb_B

    After looping all people, I wil have 2 array emb_A and emb_B, with 1680*d dimension, with d is the dimension of embedding

    To find the threshold, I do:
    - Calculate the similarity of each emb in emb_A with corresponding emb in emb_B, the result array with 1680*1680 dimension
    - Create the label array
    - Calculate the Precision, Recall and F1 Score of the similarity and the label
    - At the similarity score that has greatest F1 Score, it is the threshold
"""
)

center_block.write("Why **Cosine Similarity** and **F1 Score** are choosen?")
center_block.write(
    """
    There are many diferrent similarity methods, like Cosine Distance, Euclidean Distance, but I choose Cosine Similarity, because 2 reasons:
    - Cosine Similarity Score is in range [-1, 1], different with Euclidean Distance in range [0, 4], and no method to convert them  linearly
    - The higher Cosine Similarity, the higher similarity of 2 embedding, opposite with Cosine Distance
    - My experiment shows that the Cosine Similarity in Face Embedding is only in range [0, 1], suitable for calculating, it likes Binary Classification
"""
)

center_block.header("Results")
center_block.write("Result in LFW dataset")
center_block.image(
    load_image("./img/lfw_retina_arc_cosine_similarity_curve_center_face.png")
)
center_block.write(
    "The highest F1 Score is 0.9928, at similarity score 0.36786744, as acceptance threshold"
)
center_block.write(
    "Compare with the [ArcFace paper](https://arxiv.org/abs/1801.07698), the verification result is 0.9953 at loss function 0.5"
)
center_block.write("Why difference about the paper result and the experiment result?")
center_block.write(
    "By me: There are some faces choosen wrong by my Face detection code to get face: need to get face manually to get the accurate result"
)

center_block.header("Conclusion")
center_block.write(
    """
    By this project, I have built:
    - A full statistics about LFW dataset.
    - The complete flow to find the threshold of acceptance in LFW dataset, compare it with the public paper, and explain the wrong between them.
    - A web app to easily implement face verification of 2 pictures.
    """
)

center_block.write(
    """
    Need to improvement:
    - Implement the flow with another metrics, like Euclidean distance, ...
    - Implement the Siamese Networks to transfer learning, using non-linear approach
    - Implement on another face dataset, like WIDER Face, VGG Face, ...
    """
)


center_block.header("Citation")
center_block.write(
    "[Labeled Faces In The Wild (LFW)](http://vis-www.cs.umass.edu/lfw/)"
)
center_block.code(
    """
    @TechReport{LFWTech,
    author =       {Gary B. Huang and Manu Ramesh and Tamara Berg and 
                    Erik Learned-Miller},
    title =        {Labeled Faces in the Wild: A Database for Studying 
                    Face Recognition in Unconstrained Environments},
    institution =  {University of Massachusetts, Amherst},
    year =         2007,
    number =       {07-49},
    month =        {October}
}"""
)

center_block.write(
    "[RetinaFace: Single-stage Dense Face Localisation in the Wild](https://arxiv.org/abs/1905.00641)"
)
center_block.code(
    """
    @inproceedings{Deng2020CVPR,
    title = {RetinaFace: Single-Shot Multi-Level Face Localisation in the Wild},
    author = {Deng, Jiankang and Guo, Jia and Ververas, Evangelos and Kotsia, Irene and Zafeiriou, Stefanos},
    booktitle = {CVPR},
    year = {2020}
}"""
)
center_block.write(
    "[ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/abs/1801.07698)"
)
center_block.code(
    """
    @inproceedings{deng2018arcface,
    title={ArcFace: Additive Angular Margin Loss for Deep Face Recognition},
    author={Deng, Jiankang and Guo, Jia and Niannan, Xue and Zafeiriou, Stefanos},
    booktitle={CVPR},
    year={2019}
    }"""
)
center_block.write(
    "[Cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity)"
)
center_block.code(
    """
    @misc{ enwiki:1123118417,
    author = "{Wikipedia contributors}",
    title = "Cosine similarity --- {Wikipedia}{,} The Free Encyclopedia",
    year = "2022",
    url = "https://en.wikipedia.org/w/index.php?title=Cosine_similarity&oldid=1123118417",
    note = "[Online; accessed 6-December-2022]"
    }"""
)
center_block.write(
    "[Cosine Similarity Metric Learning for Face Verification](https://link.springer.com/chapter/10.1007/978-3-642-19309-5_55)"
)
center_block.code(
    """
    @incollection{Nguyen2011-sa,
    title     = "Cosine similarity metric learning for face verification",
    booktitle = "Computer Vision -- {ACCV} 2010",
    author    = "Nguyen, Hieu V and Bai, Li",
    publisher = "Springer Berlin Heidelberg",
    pages     = "709--720",
    series    = "Lecture notes in computer science",
    year      =  2011,
    address   = "Berlin, Heidelberg"
    }"""
)

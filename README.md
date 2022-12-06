# face_verification_demo

<div>
<a href="https://colab.research.google.com/github/LeNguyenGiaBao/face_verification_demo/blob/master/face_verification_flow.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
</div>

How does the computer know 2 pictures are the same person?

### Instructions:

Web app: [https://face-verification.streamlit.app/](https://face-verification.streamlit.app/)

Run local:

```bash
streamlit run streamlit_app.py
```

Data exploration and Statistics: Notebook [face_verification_flow.ipynb](./face_verification_flow.ipynb)

Find threshold and visualization:

```bash
python threshold.py
```

### Installation:

To install locally, please run the command:

```bash
pip install -r requirements.txt
```

### Project Overview

You are the person, and you can determine that 2 pictures are the same person, or 2 different people. So how can the machine have that ability, in the time of digization.

This application applys the archivement in Computer Vision, using Convolutional Neural Network and Cosine Similarity to verify 2 picture.

Technical Step:

- Face Detection
- Face Alignment
- Feature Extraction
- Cosine Similarity

Model:

- RetinaFace for Face Detection
- ArcFace for Face Extraction

### Data Analysis

The dataset using in this project is [Labeled Faces In The Wild (LFW)](http://vis-www.cs.umass.edu/lfw/), an offical benchmark dataset in many face verification and face recognition fields. LFW is collected, labeled and publiced in IEEE Computer Vision and Pattern Recognition 2007 by University of Massachusetts, Amherst

LFW includes **13233** images, **5749** classes, and **1680** classes have more than 2 images, image size 250x250 pixel. In a image could have many faces.

Faces in LFW is in many conditions, like positions, poses, brighness, hue, image quality, ages, genders, ...

![lfw_example](./img/lfw_example.png)

### Methodology

Data Processing:

- Get all faces in an image, include the bounding box, landmark of face by RetinaFace model
- Select the most central face in the image, using the position of bounding box of that face
- Get the embedding of the most central face
- Normalization the embedding by L2 Normalization

**Statistics for find the acceptance threshold**

With a person in 1680 people, I will:

- Get randomly 2 images of him, named img_1 and img_2
- Calculate embedding of 2 image, named emb_1 and emb_2, respectively
- Append into array emb_A and emb_B

After looping all people, I wil have 2 array emb_A and emb_B, with 1680\*d dimension, with d is the dimension of embedding

To find the threshold, I do:

- Calculate the similarity of each emb in emb_A with corresponding emb in emb_B, the result array with 1680\*1680 dimension
- Create the label array
- Calculate the Precision, Recall and F1 Score of the similarity and the label
- At the similarity score that has greatest F1 Score, it is the threshold

Why **Cosine Similarity** and **F1 Score** are choosen?

There are many diferrent similarity methods, like Cosine Distance, Euclidean Distance, but I choose Cosine Similarity, because 2 reasons:

- Cosine Similarity Score is in range [-1, 1], different with Euclidean Distance in range [0, 4], and no method to convert them linearly
- The higher Cosine Similarity, the higher similarity of 2 embedding, opposite with Cosine Distance
- My experiment shows that the Cosine Similarity in Face Embedding is only in range [0, 1], suitable for calculating, it likes Binary Classification

### Results

Result in LFW dataset

![lfw_retina_arc_cosine_similarity_curve_center_face](./img/lfw_retina_arc_cosine_similarity_curve_center_face.png)

The highest F1 Score is 0.9928, at similarity score 0.36786744, as acceptance threshold

Compare with the [ArcFace paper](https://arxiv.org/abs/1801.07698), the verification result is 0.9953 at loss function 0.5

Why difference about the paper result and the experiment result?

By me: There are some faces choosen wrong by my Face detection code to get face: need to get face manually to get the accurate result

### Conclusion

By this project, I have built:

- A full statistics about LFW dataset.
- The complete flow to find the threshold of acceptance in LFW dataset, compare it with the public paper, and explain the wrong between them.
- A web app to easily implement face verification of 2 pictures.

Need to improvement:

- Implement the flow with another metrics, like Euclidean distance, ...
- Implement the Siamese Networks to transfer learning, using non-linear approach
- Implement on another face dataset, like WIDER Face, VGG Face, ...

### Citation

```
@TechReport{LFWTech,
    author =       {Gary B. Huang and Manu Ramesh and Tamara Berg and
                    Erik Learned-Miller},
    title =        {Labeled Faces in the Wild: A Database for Studying
                    Face Recognition in Unconstrained Environments},
    institution =  {University of Massachusetts, Amherst},
    year =         2007,
    number =       {07-49},
    month =        {October}
}

@inproceedings{Deng2020CVPR,
    title = {RetinaFace: Single-Shot Multi-Level Face Localisation in the Wild},
    author = {Deng, Jiankang and Guo, Jia and Ververas, Evangelos and Kotsia, Irene and Zafeiriou, Stefanos},
    booktitle = {CVPR},
    year = {2020}
}

@inproceedings{deng2018arcface,
    title={ArcFace: Additive Angular Margin Loss for Deep Face Recognition},
    author={Deng, Jiankang and Guo, Jia and Niannan, Xue and Zafeiriou, Stefanos},
    booktitle={CVPR},
    year={2019}
}

@misc{ enwiki:1123118417,
    author = "{Wikipedia contributors}",
    title = "Cosine similarity --- {Wikipedia}{,} The Free Encyclopedia",
    year = "2022",
    url = "https://en.wikipedia.org/w/index.php?title=Cosine_similarity&oldid=1123118417",
    note = "[Online; accessed 6-December-2022]"
}

@incollection{Nguyen2011-sa,
    title     = "Cosine similarity metric learning for face verification",
    booktitle = "Computer Vision -- {ACCV} 2010",
    author    = "Nguyen, Hieu V and Bai, Li",
    publisher = "Springer Berlin Heidelberg",
    pages     = "709--720",
    series    = "Lecture notes in computer science",
    year      =  2011,
    address   = "Berlin, Heidelberg"
}
```

### Author:

<p align="center">
  <img
    src="https://avatars.githubusercontent.com/u/68860804?v=4"
    width="300px"
  />
</p>
<h2 style="text-align: center; color: black">Lê Nguyễn Gia Bảo</h2>
<p align="center">
  <a href="https://www.linkedin.com/in/lenguyengiabao/" target="_blank">
    <img src="https://img.icons8.com/fluent/48/000000/linkedin.png" />
  </a>
  <a href="https://www.facebook.com/baorua.98/" alt="Facebook" target="_blank">
    <img src="https://img.icons8.com/fluent/48/000000/facebook-new.png" />
  </a>
  <a href="https://github.com/LeNguyenGiaBao" alt="Github" target="_blank">
    <img src="https://img.icons8.com/fluent/48/000000/github.png" />
  </a>
  <a
    href="https://www.youtube.com/channel/UCOZbUfO_au3oxHEh4x52wvw/videos"
    alt="Youtube channel"
    target="_blank"
  >
    <img src="https://img.icons8.com/fluent/48/000000/youtube-play.png" />
  </a>
  <a href="https://www.kaggle.com/nguyngiabol" alt="Kaggle" target="_blank">
    <img src="https://img.icons8.com/windows/48/000000/kaggle.png" />
  </a>
  <a href="mailto:lenguyengiabao46@gmail.com" alt="Email" target="_blank">
    <img src="https://img.icons8.com/fluent/48/000000/mailing.png" />
  </a>
</p>

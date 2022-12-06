import numpy as np


def get_emb(img, face_model):
    face = face_model.get(img)
    face = sorted(
        face,
        key=lambda x: ((x["bbox"][2] - x["bbox"][0]) * (x["bbox"][3] - x["bbox"][1])),
        reverse=True,
    )[0]
    emb = face.embedding
    emb = l2_normalize(emb)

    return emb


def l2_normalize(x):
    return x / np.sqrt(np.sum(np.multiply(x, x)))


def calc_cosine_similarity(emb_A, emb_B):
    norm_emb_A = np.linalg.norm(emb_A)
    norm_emb_B = np.linalg.norm(emb_B)

    norm_emb_A = norm_emb_A.reshape(-1, 1)
    norm_emb_B = norm_emb_B.reshape(1, -1)

    mul = np.dot(emb_A, emb_B.T)
    mul_norm = np.dot(norm_emb_A, norm_emb_B)

    cosine_similarity = mul / mul_norm

    return float(cosine_similarity[0][0])

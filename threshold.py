import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve

array_A = np.load("./data/emb_data/lfw_retina_arc_center_face_a.npy")
array_B = np.load("./data/emb_data/lfw_retina_arc_center_face_b.npy")

norm_array_A = np.linalg.norm(array_A, axis=1)
norm_array_B = np.linalg.norm(array_B, axis=1)

norm_array_A = norm_array_A.reshape(-1, 1)
norm_array_B = norm_array_B.reshape(1, -1)

mul = np.dot(array_A, array_B.T)
mul_norm = np.dot(norm_array_A, norm_array_B)

cosine_similarity = mul / mul_norm
y_pred = cosine_similarity.flatten()
y_pred = np.where(y_pred > 1, 1, y_pred)
y_label = np.eye(N=cosine_similarity.shape[0], dtype=int).flatten()

precisions, recalls, thresholds = precision_recall_curve(y_label, y_pred)
f1_score = (2 * precisions * recalls) / (precisions + recalls + 1e-6)
print("Threshold shape", thresholds.shape)
threshold_index = np.argmax(f1_score)
threshold_value = thresholds[threshold_index]
precision_value = precisions[threshold_index]
recall_value = recalls[threshold_index]
f1_score_value = f1_score[threshold_index]

print(threshold_value)
plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
plt.plot(thresholds, recalls[:-1], "g--", label="Recall", linewidth=2)
plt.plot(thresholds, f1_score[:-1], "r--", label="F1 Score", linewidth=2)

# line at threshold value
plt.plot([threshold_value, threshold_value], [0, 1], "r:")

plt.text(threshold_value, 0, threshold_value, c="r")
plt.text(
    threshold_value,
    f1_score_value + 0.05,
    "F1 Score: {:.4f}".format(f1_score_value),
    c="r",
)

plt.legend()
plt.grid(True)
plt.xlabel("Threshold")
plt.title("Cosine Similarity Curve On LFW With ArcFace")
plt.savefig("./img/lfw_retina_arc_cosine_similarity_curve_center_face.png")
plt.show()

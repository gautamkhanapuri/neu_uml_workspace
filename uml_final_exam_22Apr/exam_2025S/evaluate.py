def evaluate(true_labels: np.ndarray, pred_labels: np.ndarray) -> tuple:
  """Entropy-based evaluation of a label assignment.
 
  Parameters:
    true_labels: the ground-truth class labels on the input data.
    pred_labels: the predicted class labels on the input data.
 
  Returns:
    a tuple (CM, (cs_e, cr_e, we)) containing the confusion matrix `CM`, the class entropies `cs_e`,
    the cluster entropies `cr_e`, and the averaged weighted entropies `we`.
  """
  from scipy.stats import entropy
 
  assert len(true_labels) == len(pred_labels), "Label predictions don't match"
 
  ## Map the labels to index set {0, 1, ..., k - 1 }
  t_classes, t_labels = np.unique(true_labels, return_inverse=True)
  p_classes, p_labels = np.unique(pred_labels, return_inverse=True)
  assert np.all(np.isin(p_classes, t_classes)), "Predicted class outside of labels given"
 
  ## Accumulate the counts
  n_classes = len(t_classes)
  CM = np.zeros(shape=(n_classes, n_classes), dtype=np.uint32)
  ind = np.ravel_multi_index([t_labels, p_labels], CM.shape)
  np.add.at(CM.ravel(), ind, 1)
 
  ## Compute the entropy of the empirical row/column distributions
  empirical_dist = lambda x: x / np.sum(x)
  cluster_entropy = np.apply_along_axis(lambda x: entropy(empirical_dist(x), base=2), 0, CM)
  class_entropy = np.apply_along_axis(lambda x: entropy(empirical_dist(x), base=2), 1, CM)
 
  ## Average w/ count weights
  w_cluster_entropy = np.sum(cluster_entropy * CM.sum(axis=0)) / len(y)
  w_class_entropy = np.sum(class_entropy * CM.sum(axis=1)) / len(y)
  w_entropies = np.array([w_class_entropy, w_cluster_entropy])
 
  with np.printoptions(precision=3):
    print(f"Class Entropies: {class_entropy}")
    print(f"Cluster Entropies: {cluster_entropy}")
    print(f"Weighted average entropies: {w_entropies}, (avg: {np.mean(w_entropies):.3f})")
  return CM, (w_class_entropy, w_cluster_entropy, w_entropies)
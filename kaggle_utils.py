import numpy as np
from sklearn.metrics import recall_score


def multi_task_macro_recall(true_graphemes: np.array, pred_graphemes: np.array,
                            true_vowels: np.array, pred_vowels: np.array,
                            true_consonants: np.array, pred_consonants: np.array,
                            n_grapheme=168, n_vowel=11, n_consonant=7):
    
    # pred_label_graphemes = torch.argmax(pred_graphemes, dim=1).cpu().numpy()
    # true_label_graphemes = true_graphemes.cpu().numpy()
    # pred_label_vowels = torch.argmax(pred_vowels, dim=1).cpu().numpy()
    # true_label_vowels = true_vowels.cpu().numpy()
    # pred_label_consonants = torch.argmax(pred_consonants, dim=1).cpu().numpy()
    # true_label_consonants = true_consonants.cpu().numpy()

    assert pred_graphemes.ndim == 1
    assert true_graphemes.ndim == 1
    assert pred_vowels.ndim == 1
    assert true_vowels.ndim == 1
    assert pred_consonants.ndim == 1
    assert true_consonants.ndim == 1

    grapheme_recall = recall_score(true_graphemes, pred_graphemes, labels=np.arange(n_grapheme), average='macro')
    vowel_recall = recall_score(true_vowels, pred_vowels, labels=np.arange(n_vowel), average='macro')
    consonant_recall = recall_score(true_consonants, pred_consonants, labels=np.arange(n_consonant), average='macro')
    recalls = [grapheme_recall, vowel_recall, consonant_recall]
    final_score = np.average(recalls, weights=[2, 1, 1])

    return final_score, recalls


def submit():
    pass # TODO:
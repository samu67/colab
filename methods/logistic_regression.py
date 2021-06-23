from sklearn.linear_model import LogisticRegression

from common.read_data import *


def extract_features(x):
    # extract features from each patch
    return np.concatenate([np.mean(x, (-2, -3)), np.var(x, (-2, -3))], axis=-1)


if __name__ == "__main__":
    x_train = extract_features(train_patches)
    x_val = extract_features(val_patches)

    clf = LogisticRegression(class_weight="balanced").fit(x_train, train_labels)
    print(f"Training accuracy: {clf.score(x_train, train_labels)}")
    print(f"Validation accuracy: {clf.score(x_val, val_labels)}")

    show_patched_image(train_patches[: 25 * 25], clf.predict(x_train[: 25 * 25]))

    test_path = "data/test_images/test_images"
    test_filenames = sorted(glob(test_path + "/*png"))
    test_images = load_all_from_path(test_path)
    test_patches = image_to_patches(test_images)
    x_test = extract_features(test_patches)
    test_pred = clf.predict(x_test).reshape(
        -1, test_images.shape[1] // PATCH_SIZE, test_images.shape[2] // PATCH_SIZE
    )
    create_submission(
        test_pred,
        test_filenames,
        submission_filename="data/submissions/logreg_submission.csv",
    )

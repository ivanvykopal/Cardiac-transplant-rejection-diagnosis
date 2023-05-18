import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import array_to_img


def display_sample(display_list, title=None):
    plt.figure(figsize=(18, 18))

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        if title:
            plt.title(title[i])
        plt.imshow(array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()
